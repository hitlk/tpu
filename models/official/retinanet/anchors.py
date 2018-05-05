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
"""RetinaNet anchor definition.

This module implements RetinaNet anchor described in:

T.-Y. Lin, P. Goyal, R. Girshick, K. He,  and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import tensorflow as tf
from object_detection import argmax_matcher
from object_detection import box_list
from object_detection import faster_rcnn_box_coder
from object_detection import region_similarity_calculator
from object_detection import target_assigner
from object_detection import ops
from object_detection import box_list_ops

# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -5.0

# The maximum number of (anchor,class) pairs to keep for non-max suppression.
MAX_DETECTION_POINTS = 5000


def sigmoid(x):
  """Sigmoid function for use with Numpy for CPU evaluation."""
  return 1 / (1 + np.exp(-x))


def decode_box_outputs(rel_codes, anchors):
  """Transforms relative regression coordinates to absolute positions.

  Network predictions are normalized and relative to a given anchor; this
  reverses the transformation and outputs absolute coordinates for the input
  image.

  Args:
    rel_codes: box regression targets.
    anchors: anchors on all feature levels.
  Returns:
    outputs: bounding boxes.

  """
  ycenter_a = (anchors[0] + anchors[2]) / 2
  xcenter_a = (anchors[1] + anchors[3]) / 2
  ha = anchors[2] - anchors[0]
  wa = anchors[3] - anchors[1]
  ty, tx, th, tw = rel_codes

  w = np.exp(tw) * wa
  h = np.exp(th) * ha
  ycenter = ty * ha + ycenter_a
  xcenter = tx * wa + xcenter_a
  ymin = ycenter - h / 2.
  xmin = xcenter - w / 2.
  ymax = ycenter + h / 2.
  xmax = xcenter + w / 2.
  return np.column_stack([ymin, xmin, ymax, xmax])


def nms(dets, thresh):
  """Non-maximum supression."""
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 4]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    intersection = w * h
    overlap = intersection / (areas[i] + areas[order[1:]] - intersection)

    inds = np.where(overlap <= thresh)[0]
    order = order[inds + 1]
  return keep


def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
  """Generates mapping from output level to a list of anchor configurations.

  A configuration is a tuple of (num_anchors, scale, aspect_ratio).

  Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect raito anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
  Returns:
    anchor_configs: a dictionary with keys as the levels of anchors and
      values as a list of anchor configuration.
  """
  anchor_configs = {}
  for level in range(min_level, max_level + 1):
    anchor_configs[level] = []
    for scale_octave in range(num_scales):
      for aspect in aspect_ratios:
        anchor_configs[level].append(
            (2**level, scale_octave / float(num_scales), aspect))
  return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
  """Generates multiscale anchor boxes.

  Args:
    image_size: tuple of (height, width). The image_size should be divided by
      the largest feature stride 2^max_level.
    anchor_scale: float number representing the scale of size of the base
      anchor to the feature stride 2^level.
    anchor_configs: a dictionary with keys as the levels of anchors and
      values as a list of anchor configuration.
  Returns:
    anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
      feature levels.
  Raises:
    ValueError: input size must be the multiple of largest feature stride.
  """
  boxes_all = []
  for _, configs in anchor_configs.items():
    boxes_level = []
    for config in configs:
      stride, octave_scale, aspect = config
      if image_size[0] % stride != 0 or image_size[1] % stride != 0:
        raise ValueError("input size must be divided by the stride.")
      base_anchor_size = anchor_scale * stride * 2**octave_scale
      anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
      anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0

      x = np.arange(stride / 2, image_size[1], stride)
      y = np.arange(stride / 2, image_size[0], stride)
      xv, yv = np.meshgrid(x, y)
      xv = xv.reshape(-1)
      yv = yv.reshape(-1)

      boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                         yv + anchor_size_y_2, xv + anchor_size_x_2))
      boxes = np.swapaxes(boxes, 0, 1)
      boxes_level.append(np.expand_dims(boxes, axis=1))
    # concat anchors on the same level to the reshape NxAx4
    boxes_level = np.concatenate(boxes_level, axis=1)
    boxes_all.append(boxes_level.reshape([-1, 4]))

  anchor_boxes = np.vstack(boxes_all)
  return anchor_boxes


def _generate_detections(cls_outputs, box_outputs, anchor_boxes, image_id):
  """Generates detections with RetinaNet model outputs and anchors.

  Args:
    cls_outputs: a numpy array with shape [N, num_classes], which stacks class
      logit outputs on all feature levels. The N is the number of total anchors
      on all levels. The num_classes is the number of classes predicted by the
      model.
    box_outputs: a numpy array with shape [N, 4], which stacks box regression
      outputs on all feature levels. The N is the number of total anchors on all
      levels.
    anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
      feature levels. The N is the number of total anchors on all levels.
    image_id: an integer number to specify the image id.
  Returns:
    detections: detection results in a tensor with each row representing
      [image_id, x, y, width, height, score, class]
  """
  num_classes = cls_outputs.shape[1]
  cls_outputs_reshape = cls_outputs.reshape(-1)
  # speed up inference by filtering out low scoring logits
  indices = np.where(cls_outputs_reshape > MIN_CLASS_SCORE)[0]
  if indices.any():
    length = min(len(indices), MAX_DETECTION_POINTS)
    # use argpartition instead of argsort to speed up
    indices_top_k = np.argpartition(
        -cls_outputs_reshape[indices], length-1)[0:length-1]
    indices_top_k = indices[indices_top_k]
  else:
    indices_top_k = np.argsort(-cls_outputs_reshape)[0:MAX_DETECTION_POINTS]

  indices, classes = np.unravel_index(indices_top_k, cls_outputs.shape)
  anchor_boxes = anchor_boxes[indices, :]
  box_outputs = box_outputs[indices, :]
  scores = sigmoid(cls_outputs[indices, classes])
  # apply bounding box regression to anchors
  boxes = decode_box_outputs(
      box_outputs.swapaxes(0, 1), anchor_boxes.swapaxes(0, 1))
  boxes = boxes[:, [1, 0, 3, 2]]
  # run class-wise nms
  detections = []
  for c in range(num_classes):
    indices = np.where(classes == c)[0]
    if indices.shape[0] == 0:
      continue
    boxes_cls = boxes[indices, :]
    scores_cls = scores[indices]
    # Select top-scoring boxes in each class and apply non-maximum suppression
    # (nms) for boxes in the same class. The selected boxes from each class are
    # then concatenated for the final detection outputs.
    all_detections_cls = np.column_stack((boxes_cls, scores_cls))
    top_detection_idx = nms(all_detections_cls, 0.5)
    top_detections_cls = all_detections_cls[top_detection_idx]
    top_detections_cls[:, 2] -= top_detections_cls[:, 0]
    top_detections_cls[:, 3] -= top_detections_cls[:, 1]
    top_detections_cls = np.column_stack(
        (np.repeat(image_id, len(top_detection_idx)),
         top_detections_cls,
         np.repeat(c + 1, len(top_detection_idx)))
    )
    detections.append(top_detections_cls)

  detections = np.vstack(detections)
  # take final 100 detections
  indices = np.argsort(-detections[:, -2])
  detections = np.array(detections[indices[0:100]], dtype=np.float32)
  return detections

def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):
  """Create a tiled set of anchors strided along a grid in image space.

  This op creates a set of anchor boxes by placing a "basis" collection of
  boxes with user-specified scales and aspect ratios centered at evenly
  distributed points along a grid.  The basis collection is specified via the
  scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]
  and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
  .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
  and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
  placing it over its respective center.

  Grid points are specified via grid_height, grid_width parameters as well as
  the anchor_stride and anchor_offset parameters.

  Args:
    grid_height: size of the grid in the y direction (int or int scalar tensor)
    grid_width: size of the grid in the x direction (int or int scalar tensor)
    scales: a 1-d  (float) tensor representing the scale of each box in the
      basis set.
    aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
      box in the basis set.  The length of the scales and aspect_ratios tensors
      must be equal.
    base_anchor_size: base anchor size as [height, width]
      (float tensor of shape [2])
    anchor_stride: difference in centers between base anchors for adjacent grid
                   positions (float tensor of shape [2])
    anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                   upper left element of the grid, this should be zero for
                   feature networks with only VALID padding and even receptive
                   field size, but may need some additional calculation if other
                   padding is used (float tensor of shape [2])
  Returns:
    a BoxList holding a collection of N anchor boxes
  """
  ratio_sqrts = tf.sqrt(aspect_ratios)
  heights = scales / ratio_sqrts * base_anchor_size[0]
  widths = scales * ratio_sqrts * base_anchor_size[1]

  # Get a grid of box centers
  y_centers = tf.to_float(tf.range(grid_height))
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = tf.to_float(tf.range(grid_width))
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = ops.meshgrid(x_centers, y_centers)

  widths_grid, x_centers_grid = ops.meshgrid(widths, x_centers)
  heights_grid, y_centers_grid = ops.meshgrid(heights, y_centers)
  bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
  bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
  bbox_centers = tf.reshape(bbox_centers, [-1, 2])
  bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
  bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
  return box_list.BoxList(bbox_corners)

def _center_size_bbox_to_corners_bbox(centers, sizes):
  """Converts bbox center-size representation to corners representation.

  Args:
    centers: a tensor with shape [N, 2] representing bounding box centers
    sizes: a tensor with shape [N, 2] representing bounding boxes

  Returns:
    corners: tensor with shape [N, 4] representing bounding boxes in corners
      representation
  """
  return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)


class Anchors(object):
  """RetinaNet Anchors class."""

  def __init__(self, min_level, max_level, num_scales, aspect_ratios,
               anchor_scale, image_size):
    """Constructs multiscale RetinaNet anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect raito anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
      anchor_scale: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: tuple of [height, width]. The image_size should be divided by
        the largest feature stride 2^max_level.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_scale = anchor_scale
    self.image_size = image_size
    self.config = self._generate_configs()
    # self.boxes = self._generate_boxes()
    self.boxes = self._generate()
  def _generate_configs(self):
    """Generate configurations of anchor boxes."""
    return _generate_anchor_configs(self.min_level, self.max_level,
                                    self.num_scales, self.aspect_ratios)

  def _generate_boxes(self):
    """Generates multiscale anchor boxes."""
    boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale,
                                   self.config)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    return boxes

  def _generate(self):
    im_height, im_width = self.image_size
    aspect_ratios = [w / h for (h, w) in self.aspect_ratios]
    num_scales = self.num_scales
    scales = [scale_octave / float(num_scales) for scale_octave in range(num_scales)]

    anchors_list = []
    for level in (self.min_level, self.max_level + 1):
      stride = 2**level
      grid_height = im_height / stride
      grid_width = im_width / stride
      base_anchor_size = [self.anchor_scale * stride, self.anchor_scale * stride]
      anchor_stride = [stride, stride]
      anchor_offset = [stride / 2, stride / 2]

      scales_grid, aspect_ratios_grid = ops.meshgrid(scales,
                                                     aspect_ratios)
      scales_grid = tf.reshape(scales_grid, [-1])
      aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
      anchors = tile_anchors(grid_height,
                             grid_width,
                             scales_grid,
                             aspect_ratios_grid,
                             base_anchor_size,
                             anchor_stride,
                             anchor_offset)
      anchors_list.append(anchors)

    return box_list_ops.concatenate(anchors_list)

  def get_anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)


class AnchorLabeler(object):
  """Labeler for multiscale anchor boxes."""

  def __init__(self, anchors, num_classes, match_threshold=0.5, unmatched_threshold=0.4):
    """Constructs anchor labeler to assign labels to anchors.

    Args:
      anchors: an instance of class Anchors.
      num_classes: integer number representing number of classes in the dataset.
      match_threshold: float number between 0 and 1 representing the threshold
        to assign positive labels for anchors.
    """
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(
        match_threshold,
        unmatched_threshold=unmatched_threshold,
        negatives_lower_than_unmatched=True,
        force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

    self._target_assigner = target_assigner.TargetAssigner(
        similarity_calc, matcher, box_coder)
    self._anchors = anchors
    self._match_threshold = match_threshold
    self._num_classes = num_classes

  def _unpack_labels(self, labels):
    """Unpacks an array of labels into multiscales labels."""
    labels_unpacked = OrderedDict()
    anchors = self._anchors
    count = tf.constant(0, dtype=tf.int32)
    for level in range(anchors.min_level, anchors.max_level + 1):
      # feat_size = int(anchors.image_size / 2**level)
      grid_height, grid_width = anchors.image_size
      grid_height = tf.floor(grid_height / 2**level)
      grid_width = tf.floor(grid_width / 2**level)
      steps = (grid_height * grid_width) * anchors.get_anchors_per_location()
      steps = tf.cast(steps, tf.int32)
      indices = tf.range(count, count + steps)
      count = count + steps
      labels_unpacked[level] = tf.reshape(
          tf.gather(labels, indices), [grid_height, grid_width, -1])
    return labels_unpacked

  def label_anchors(self, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
      gt_labels: A integer tensor with shape [N, 1] representing groundtruth
        classes.
    Returns:
      cls_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of class logits at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
      num_positives: scalar tensor storing number of positives in an image.
    """
    gt_box_list = box_list.BoxList(gt_boxes)
    # anchor_box_list = box_list.BoxList(self._anchors.boxes)
    anchor_box_list = self._anchors.boxes

    # cls_weights, box_weights are not used
    cls_targets, cls_weights, box_targets, box_weights, matches = self._target_assigner.assign(
        anchor_box_list, gt_box_list, gt_labels)

    # class labels start from 1 and the background class = -1
    cls_targets -= 1
    cls_targets = tf.cast(cls_targets, tf.int32)

    # Unpack labels.
    cls_targets_dict = self._unpack_labels(cls_targets)
    cls_weights_dict = self._unpack_labels(cls_weights)
    box_targets_dict = self._unpack_labels(box_targets)
    box_weights_dict = self._unpack_labels(box_weights)

    num_positives = tf.reduce_sum(
        tf.cast(tf.greater(matches.match_results, -1), tf.float32))

    return cls_targets_dict, cls_weights_dict, box_targets_dict, box_weights_dict, \
           num_positives

  def generate_detections(self, cls_ouputs, box_outputs, image_id):
    cls_outputs_all = []
    box_outputs_all = []
    for level in range(self._anchors.min_level, self._anchors.max_level + 1):
      cls_outputs_all.append(
          tf.reshape(cls_ouputs[level], [-1, self._num_classes]))
      box_outputs_all.append(tf.reshape(box_outputs[level], [-1, 4]))
    cls_outputs_all = tf.concat(cls_outputs_all, 0)
    box_outputs_all = tf.concat(box_outputs_all, 0)
    return tf.py_func(
        _generate_detections,
        [cls_outputs_all, box_outputs_all, self._anchors.boxes, image_id],
        tf.float32)
