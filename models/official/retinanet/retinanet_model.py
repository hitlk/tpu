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
"""Model defination for the RetinaNet Model.

Defines model_fn of RetinaNet for TF Estimator. The model_fn includes RetinaNet
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import anchors
import coco_metric
import retinanet_architecture
from tensorflow import estimator
# from tensorflow.contrib.tpu.python.tpu import bfloat16
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.framework import filter_variables

# A collection of Learning Rate schecules:
# third_party/tensorflow_models/object_detection/utils/learning_schedules.py
def _learning_rate_schedule(base_learning_rate, lr_warmup_init, lr_warmup_step,
                            lr_drop_step, global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # lr_warmup_init is the starting learning rate; the learning rate is linearly
  # scaled up to the full learning rate after `lr_warmup_steps` before decaying.
  lr_warmup_remainder = 1.0 - lr_warmup_init
  linear_warmup = [
      (lr_warmup_init + lr_warmup_remainder * (float(step) / lr_warmup_step),
       step) for step in range(0, lr_warmup_step, max(1, lr_warmup_step // 100))
  ]
  lr_schedule = linear_warmup + [[1.0, lr_warmup_step], [0.1, lr_drop_step], [0.01, lr_drop_step + 80000]]
  learning_rate = base_learning_rate
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             base_learning_rate * mult)
  return learning_rate


def focal_loss(logits, targets, alpha, gamma, normalizer):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-alpha)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    logits: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    targets: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.
    normalizer: A float32 scalar normalizes the total loss from all examples.
  Returns:
    loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    probs = tf.sigmoid(logits)
    probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
    # With small gamma, the implementation could produce NaN during back prop.
    modulator = tf.pow(1.0 - probs_gt, gamma)
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    total_loss = tf.reduce_sum(weighted_loss)
    total_loss /= normalizer
  return total_loss


def _classification_loss(cls_outputs,
                         cls_targets,
                         num_positives,
                         alpha=0.25,
                         gamma=2.0):
  """Computes classification loss."""
  normalizer = num_positives
  classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma,
                                   normalizer)
  return classification_loss


def _cls_loss(prediction_tensor,
              target_tensor,
              weights,
              num_positives,
              alpha=0.25,
              gamma=2.0):
  """Compute loss function.
  Args:
    prediction_tensor: A float tensor of size [batch_size, num_anchors, num_classes] representing
      the predicted logits for each class.
    target_tensor: A float tensor of size [batch_size, num_anchors, num_classes] representing
      on-hot encoded classification targets.
    weights: A float tensor of size [batch_size, num_anchors].
  """
  normalizer = num_positives
  weights = tf.expand_dims(weights, axis=2)
  per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
    labels=target_tensor, logits=prediction_tensor))
  prediction_probabilities = tf.sigmoid(prediction_tensor)
  p_t = ((target_tensor * prediction_probabilities) +
         ((1 - target_tensor) * (1 - prediction_probabilities)))
  modulating_factor = 1.0
  if gamma:
    modulating_factor = tf.pow(1.0 - p_t, gamma)
  alpha_weight_factor = 1.0
  if alpha is not None:
    alpha_weight_factor = (target_tensor * alpha +
                           (1 - target_tensor) * (1 - alpha))
  focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                              per_entry_cross_ent)
  total_loss = tf.reduce_sum(focal_cross_entropy_loss * weights)
  total_loss /= normalizer

  return total_loss


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
  normalizer = num_positives * 4.0
  mask = tf.not_equal(box_targets, 0.0)
  box_loss = tf.losses.huber_loss(
      box_targets,
      box_outputs,
      weights=mask,
      delta=delta,
      reduction=tf.losses.Reduction.SUM)
  box_loss /= normalizer
  return box_loss

def _bbox_loss(prediction_tensor, target_tensor, weights, num_positives, delta=0.1):
  """Compute loss function.

  Args:
    prediction_tensor: A float tensor of shape [batch_size, num_anchors,
      code_size] representing the (encoded) predicted locations of objects.
    target_tensor: A float tensor of shape [batch_size, num_anchors,
      code_size] representing the regression targets
    weights: a float tensor of shape [batch_size, num_anchors]

  Returns:
    loss: a float tensor of shape [batch_size, num_anchors] tensor
      representing the value of the loss function.
  """
  normalizer = num_positives
  box_loss = tf.losses.huber_loss(
    target_tensor,
    prediction_tensor,
    delta=delta,
    weights=tf.expand_dims(weights, axis=2),
    loss_collection=None,
    reduction=tf.losses.Reduction.NONE
  )
  box_loss = tf.reduce_sum(box_loss)
  box_loss /= normalizer

  return box_loss

def _detection_loss(cls_outputs, box_outputs, labels, params):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in
      [batch_size, height, width, num_anchors * num_classes].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_loss: an integar tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: an integar tensor representing total class loss.
    box_loss: an integar tensor representing total box regression loss.
  """
  # Sum all positives in a batch for normalization and avoid zero
  # num_positives_sum, which would lead to inf loss during training
  num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
  num_negatives_sum = tf.reduce_sum(labels['mean_num_negatives'])
  num_ignored_sum = tf.reduce_sum(labels['mean_num_ignored'])

  # summary
  tf.summary.scalar('num_positives', num_positives_sum)
  tf.summary.scalar('num_negatives', num_negatives_sum)
  tf.summary.scalar('num_ignored', num_ignored_sum)

  levels = cls_outputs.keys()

  cls_losses = []
  box_losses = []
  for level in levels:
    # Onehot encoding for classification labels.
    cls_targets_at_level = tf.one_hot(
        labels['cls_targets_%d' % level],
        params['num_classes'])
    bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
    # cls_targets_at_level = tf.reshape(cls_targets_at_level,
    #                                    [bs, width, height, -1])
    # cls_losses.append(
    #     _classification_loss(
    #         cls_outputs[level],
    #         cls_targets_at_level,
    #         num_positives_sum,
    #         alpha=params['alpha'],
    #         gamma=params['gamma']))
    cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                     [bs, -1, params['num_classes']])
    cls_outputs_at_level = cls_outputs[level]
    cls_outputs_at_level = tf.reshape(cls_outputs_at_level,
                                      [bs, -1, params['num_classes']])
    cls_weights_at_level = labels['cls_weights_%d' % level]
    cls_weights_at_level = tf.reshape(cls_weights_at_level,
                                      [bs, -1])
    tf.summary.scalar('cls_weights_%d' % level, tf.reduce_sum(cls_weights_at_level))
    cls_losses.append(
      _cls_loss(
        cls_outputs_at_level,
        cls_targets_at_level,
        cls_weights_at_level,
        num_positives_sum))

    box_targets_at_level = labels['box_targets_%d' % level]
    box_targets_at_level = tf.reshape(box_targets_at_level,
                                      [bs, -1, 4])
    box_outputs_at_level = box_outputs[level]
    box_outputs_at_level = tf.reshape(box_outputs_at_level,
                                      [bs, -1, 4])
    box_weights_at_level = labels['box_weights_%d' % level]
    box_weights_at_level = tf.reshape(box_weights_at_level,
                                      [bs, -1])
    tf.summary.scalar('box_weight_%d' % level, tf.reduce_sum(box_weights_at_level))
    box_losses.append(
        _bbox_loss(
          box_outputs_at_level,
          box_targets_at_level,
          box_weights_at_level,
          num_positives_sum,
          params['delta']))

  # Sum per level losses to total loss.
  for i, loss in enumerate(cls_losses):
    tf.summary.scalar('fl_fpn{}'.format(i + 3), loss)
  for i, loss in enumerate(box_losses):
    tf.summary.scalar('bbox_loss_fpn{}'.format(i + 3), loss)
  cls_loss = tf.add_n(cls_losses)
  box_loss = tf.add_n(box_losses)
  cls_loss = tf.check_numerics(cls_loss, 'cls_loss is inf or nan.')
  box_loss = tf.check_numerics(box_loss, 'box_loss is inf or nan.')
  total_loss = cls_loss + params['box_loss_weight'] * box_loss
  return total_loss, cls_loss, box_loss


def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model definition for the RetinaNet model based on ResNet.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the RetinaNet model outputs class logits and box regression outputs.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
  """
  def _model_outputs():
    return model(
        features,
        min_level=params['min_level'],
        max_level=params['max_level'],
        num_classes=params['num_classes'],
        num_anchors=len(params['aspect_ratios'] * params['num_scales']),
        resnet_depth=params['resnet_depth'],
        is_training_bn=params['is_training_bn'])

  # if params['use_bfloat16']:
  #   with bfloat16.bfloat16_scope():
  #     cls_outputs, box_outputs = _model_outputs()
  #     levels = cls_outputs.keys()
  #     for level in levels:
  #       cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
  #       box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  # else:
  #   cls_outputs, box_outputs = _model_outputs()
  #   levels = cls_outputs.keys()
  cls_outputs, box_outputs = _model_outputs()
  levels = cls_outputs.keys()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'image': features,
    }
    for level in levels:
      predictions['cls_outputs_%d' % level] = cls_outputs[level]
      predictions['box_outputs_%d' % level] = box_outputs[level]
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Load pretrained model from checkpoint.
  if params['resnet_checkpoint'] and mode == tf.estimator.ModeKeys.TRAIN:

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      # tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
      #     '/': 'resnet%s/' % params['resnet_depth'],
      # })
      variables_to_restore = {}
      for variable in tf.global_variables():
        variable_name = variable.op.name
        if variable_name.startswith('resnet50') and 'group_norm' not in variable_name:
          var_name = variable_name.replace('resnet50/', '')
          variables_to_restore[var_name] = variable
      init_saver = tf.train.Saver(variables_to_restore)
      def init_fn(scaffold, sess):
        init_saver.restore(sess, params['resnet_checkpoint'])
      return tf.train.Scaffold(init_fn=init_fn)
  elif params['fine_tune_checkpoint'] and mode == tf.estimator.ModeKeys.TRAIN:
    def scaffold_fn():
      variable_to_restore = tf.global_variables()
      variable_to_restore = filter_variables(variable_to_restore,
                                             exclude_patterns=['class-predict', 'box-predict', 'Momentum'])
      init_saver = tf.train.Saver(variable_to_restore)
      def init_fn(scaffold, sess):
        init_saver.restore(sess, params['fine_tune_checkpoint'])
      return tf.train.Scaffold(init_fn=init_fn)
  else:
    scaffold_fn = None

  # Set up training loss and learning rate.
  global_step = tf.train.get_global_step()
  learning_rate = _learning_rate_schedule(
      params['learning_rate'], params['lr_warmup_init'],
      params['lr_warmup_step'], params['lr_drop_step'], global_step)
  tf.summary.scalar('learning_rate', learning_rate)
  # cls_loss and box_loss are for logging. only total_loss is optimized.
  total_loss, cls_loss, box_loss = _detection_loss(cls_outputs, box_outputs,
                                                   labels, params)

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf.summary.scalar('Losses/cls_loss', cls_loss)
    tf.summary.scalar('Losses/box_loss', box_loss)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])

    # if params['use_tpu']:
    #   optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    var_list = variable_filter_fn(
        tf.trainable_variables(),
        params['resnet_depth']) if variable_filter_fn else None
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(total_loss, global_step, var_list=var_list)
  else:
    train_op = None

  # Evaluation only works on GPU/CPU host and batch_size=1
  # eval_metrics = None
  eval_metric_ops = None
  if mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(**kwargs):
      """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""
      eval_anchors = anchors.Anchors(params['min_level'],
                                     params['max_level'],
                                     params['num_scales'],
                                     params['aspect_ratios'],
                                     params['anchor_scale'],
                                     params['image_size'])
      anchor_labeler = anchors.AnchorLabeler(eval_anchors,
                                             params['num_classes'])
      cls_loss = tf.metrics.mean(kwargs['cls_loss_repeat'])
      box_loss = tf.metrics.mean(kwargs['box_loss_repeat'])
      # add metrics to output
      cls_outputs = {}
      box_outputs = {}
      for level in range(params['min_level'], params['max_level'] + 1):
        cls_outputs[level] = kwargs['cls_outputs_%d' % level]
        box_outputs[level] = kwargs['box_outputs_%d' % level]
      detections = anchor_labeler.generate_detections(
          cls_outputs, box_outputs, kwargs['source_ids'])
      eval_metric = coco_metric.EvaluationMetric(params['val_json_file'])
      coco_metrics = eval_metric.estimator_metric_fn(detections,
                                                     kwargs['image_scales'])
      # Add metrics to output.
      output_metrics = {
          'cls_loss': cls_loss,
          'box_loss': box_loss,
      }
      output_metrics.update(coco_metrics)
      return output_metrics

    batch_size = 1
    cls_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(cls_loss, 0), [
            batch_size,
        ]), [batch_size, 1])
    box_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(box_loss, 0), [
            batch_size,
        ]), [batch_size, 1])
    metric_fn_inputs = {
        'cls_loss_repeat': cls_loss_repeat,
        'box_loss_repeat': box_loss_repeat,
        'source_ids': labels['source_ids'],
        'image_scales': labels['image_scales'],
    }
    for level in range(params['min_level'], params['max_level'] + 1):
      metric_fn_inputs['cls_outputs_%d' % level] = cls_outputs[level]
      metric_fn_inputs['box_outputs_%d' % level] = box_outputs[level]
    # eval_metrics = (metric_fn, metric_fn_inputs)
    eval_metric_ops = metric_fn(**metric_fn_inputs)

  # return tpu_estimator.TPUEstimatorSpec(
  #     mode=mode,
  #     loss=total_loss,
  #     train_op=train_op,
  #     eval_metrics=eval_metrics,
  #     scaffold_fn=scaffold_fn)
  return estimator.EstimatorSpec(
    mode=mode,
    loss=total_loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops,
    scaffold=scaffold_fn() if scaffold_fn is not None else None
  )


def retinanet_model_fn(features, labels, mode, params):
  """RetinaNet model."""
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=retinanet_architecture.retinanet,
      variable_filter_fn=retinanet_architecture.remove_variables)


def default_hparams():
  return tf.contrib.training.HParams(
      image_size=640,
      input_rand_hflip=True,
      # dataset specific parameters
      num_classes=90,
      skip_crowd=True,
      # model architecture
      min_level=3,
      max_level=7,
      num_scales=3,
      aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
      anchor_scale=4.0,
      resnet_depth=50,
      # is batchnorm training mode
      is_training_bn=True,
      # optimization
      momentum=0.9,
      learning_rate=0.08,
      lr_warmup_init=0.1,
      lr_warmup_step=2000,
      lr_drop_step=15000,
      # classification loss
      alpha=0.25,
      gamma=1.5,
      # localization loss
      delta=0.1,
      box_loss_weight=50.0,
      # resnet checkpoint
      resnet_checkpoint=None,
      # output detection
      box_max_detected=100,
      box_iou_threshold=0.5,
      use_bfloat16=False,
  )
