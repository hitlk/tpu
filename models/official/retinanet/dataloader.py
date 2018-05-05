# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data loader and processing.

Defines input_fn of RetinaNet for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

T.-Y. Lin, P. Goyal, R. Girshick, K. He,  and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

import tensorflow as tf
import numpy as np

import anchors
from object_detection import preprocessor
from object_detection import tf_example_decoder
from object_detection import shape_utils


_ASPECT_RATIO = 1.666
_COARSEST_STRIDE = 128

def _normalize_image(image):
  """Normalize the image to zero mean and unit variance."""
  offset = tf.constant([0.485, 0.456, 0.406])
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= offset

  scale = tf.constant([0.229, 0.224, 0.225])
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= scale
  return image


class InputReader(object):
  """Input reader for the MSCOCO dataset."""

  def __init__(self, file_pattern, batch_size, is_training):
    self._file_pattern = file_pattern
    self._is_training = is_training
    self._batch_size = batch_size

  def _get_feature_map_spatial_dims(self, feature_maps):
    feature_map_shapes = [
      shape_utils.combined_static_and_dynamic_shape(
        feature_map) for feature_map in feature_maps
    ]

    return [(shape[0], shape[1]) for shape in feature_map_shapes]

  def __call__(self, params):
    # input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
    #                                 params['num_scales'],
    #                                 params['aspect_ratios'],
    #                                 params['anchor_scale'],
    #                                 params['image_size'])
    # anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
    example_decoder = tf_example_decoder.TfExampleDecoder()

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets."""
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        source_id = data['source_id']
        image = data['image']
        boxes = data['groundtruth_boxes']
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        # Handle crowd annotations. As crowd annotations are not large
        # instances, the model ignores them in training.
        if params['skip_crowd']:
          indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
          classes = tf.gather_nd(classes, indices)
          boxes = tf.gather_nd(boxes, indices)

        # the image normalization is identical to Cloud TPU ResNet-50
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = _normalize_image(image)

        if params['input_rand_hflip']:
          image, boxes = preprocessor.random_horizontal_flip(image, boxes=boxes)
        image_original_shape = tf.shape(image)
        max_size = params['image_size'] * _ASPECT_RATIO
        image, new_size = preprocessor.resize_to_range(
            image,
            min_dimension=params['image_size'],
            max_dimension=max_size)
        image_scale = tf.to_float(image_original_shape[0]) / tf.to_float(
            tf.shape(image)[0])
        image, boxes = preprocessor.scale_boxes_to_pixel_coordinates(
            image, boxes, keypoints=None)
        stride = tf.to_float(_COARSEST_STRIDE)
        new_size = tf.ceil(tf.to_float(new_size) / stride) * stride
        new_size = tf.cast(new_size, tf.int32)
        image = tf.image.pad_to_bounding_box(image, 0, 0, new_size[0],
                                             new_size[1])
        # (cls_targets, cls_weights, box_targets, box_weights,
        #  num_positives, num_negatives, num_ignored) = anchor_labeler.label_anchors(boxes, classes)

        source_id = tf.string_to_number(source_id, out_type=tf.float32)
        if params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)
        # row = (image, cls_targets, cls_weights, box_targets, box_weights, num_positives, num_negatives, num_ignored,
        #        source_id, image_scale)
        row = (image, source_id, image_scale, boxes, classes)
        return row

    # batch_size = params['batch_size']
    batch_size = self._batch_size

    dataset = tf.data.Dataset.list_files(self._file_pattern)

    dataset = dataset.shuffle(buffer_size=1024)
    if self._is_training:
      dataset = dataset.repeat()

    def prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename, buffer_size=8 * 1000 * 1000)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            prefetch_dataset, cycle_length=1, sloppy=True))
    dataset = dataset.shuffle(buffer_size=3072)

    dataset = dataset.map(_dataset_parser, num_parallel_calls=12)
    dataset = dataset.prefetch(32)
    dataset = dataset.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(batch_size, ([None, None, None],
                                                                     [], [],
                                                                     [None, None],
                                                                     [None])))
    dataset = dataset.prefetch(2)

    # (images, cls_targets, cls_weights, box_targets, box_weights, num_positives, num_negatives, num_ignored, source_ids,
    #  image_scales) = dataset.make_one_shot_iterator().get_next()
    (images, source_ids, image_scales, gt_boxes, gt_classes) = dataset.make_one_shot_iterator().get_next()

    feature_map_spatial_dims = self._get_feature_map_spatial_dims(tf.unstack(images))

    cls_targets_dict = {}
    cls_weights_dict = {}
    reg_targets_dict = {}
    reg_weights_dict = {}

    num_positives_list = []

    gt_boxes_batch = tf.unstack(gt_boxes)
    gt_classes_batch = tf.unstack(gt_classes)
    gt_weights_batch = [None] * len(gt_classes_batch)
    def merge_dict(result, target):
      for level in range(params['min_level'], params['max_level'] + 1):
        if level not in result.keys():
          result[level] = []
        result[level].append(target[level])

    for gt_boxes, gt_classes, gt_weights in zip(gt_boxes_batch, gt_classes_batch, gt_weights_batch):
      input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                      params['num_scales'], params['aspect_ratios'],
                                      params['anchor_scale'], feature_map_spatial_dims[0])
      anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
      anchor_labeler.label_anchors(gt_boxes, gt_classes)

      cls_targets_single, cls_weights_single, reg_targets_single, reg_weights_single, num_positives_single \
        = anchor_labeler.label_anchors(gt_boxes, gt_classes)
      merge_dict(cls_targets_dict, cls_targets_single)
      merge_dict(cls_weights_dict, cls_weights_single)
      merge_dict(reg_targets_dict, reg_targets_single)
      merge_dict(reg_weights_dict, reg_weights_single)
      num_positives_list.append(num_positives_single)

    num_positives = tf.stack(num_positives_list)

    labels = {}
    # count num_positives in a batch
    num_positives_batch = tf.reduce_mean(num_positives)
    labels['mean_num_positives'] = tf.reshape(
        tf.tile(tf.expand_dims(num_positives_batch, 0), [
            batch_size,
        ]), [batch_size, 1])


    for level in range(params['min_level'], params['max_level'] + 1):
      labels['cls_targets_%d' % level] = tf.stack(cls_targets_dict[level])
      labels['cls_weights_%d' % level] = tf.stack(cls_weights_dict[level])
      labels['box_targets_%d' % level] = tf.stack(reg_targets_dict[level])
      labels['box_weights_%d' % level] = tf.stack[reg_weights_dict[level]]
    labels['source_ids'] = source_ids
    labels['image_scales'] = image_scales
    return images, labels


if __name__ == '__main__':
  reader_fn = InputReader('/data/coco/coco_train.record', 8, True)
  params = {
    'min_level': 3,
    'max_level': 7,
    'num_scales': 3,
    'aspect_ratios': [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
    'anchor_scale': 4,
    'image_size': 512,
    'num_classes': 90,
    'skip_crowd': True,
    'input_rand_hflip': True,
    'use_bfloat16': False
  }

  images, labels = reader_fn(params)

  output_tensor = {}
  for level in range(3, 8):
    output_tensor['box_weights_%d' % level] = tf.reduce_sum(labels['box_weights_%d' % level])
  with tf.Session() as sess:
    for i in range(30):
      print(sess.run(output_tensor))
