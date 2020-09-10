# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os, json, datetime, math

import re
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from absl import flags
#from official.utils.flags import core as flags_core


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
                     poly_power, start_warmup_step, weight_decay_input):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=poly_power,
      cycle=False)

  # Implements linear warmup. I.e., if global_step - start_warmup_step <
  # num_warmup_steps, the learning rate will be
  # `(global_step - start_warmup_step)/num_warmup_steps * init_lr`.
  # if num_warmup_steps > 0:
  # condition = tf.greater(num_warmup_steps, 0.0)
  # if condition:
  tf.logging.info("++++++ warmup starts at step " + str(start_warmup_step) + ", for " + str(num_warmup_steps) + " steps ++++++")
  global_steps_int = tf.cast(global_step, tf.int32)
  start_warm_int = tf.constant(start_warmup_step, dtype=tf.int32)
  global_steps_int = global_steps_int - start_warm_int
  warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
  global_steps_float = tf.cast(global_steps_int, tf.float32)
  warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
  warmup_percent_done = global_steps_float / warmup_steps_float
  warmup_learning_rate = init_lr * warmup_percent_done
  is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
  learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is OK that you use this optimizer for finetuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  # It is OK to use AdamW in the finetuning even the model is trained by LAMB.
  # As report in the Bert pulic github, the learning rate for SQuAD 1.1 is 3e-5,
  # 4e-5 or 5e-5. For LAMB, the users can use 3e-4, 4e-4,or 5e-4 for a batch
  # size of 64 in the finetune.
  optimizer = LAMBOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=weight_decay_input,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "batch_normalization", "BatchNormalization", "bias"])

  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `LAMBOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


class LAMBOptimizer(tf.train.Optimizer):
  """LAMB (Layer-wise Adaptive Moments optimizer for Batch training)."""
  # A new optimizer that includes correct L2 weight decay, adaptive
  # element-wise updating, and layer-wise justification. The LAMB optimizer
  # was proposed by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song,
  # James Demmel, and Cho-Jui Hsieh in a paper titled as Reducing BERT
  # Pre-Training Time from 3 Days to 76 Minutes (arxiv.org/abs/1904.00962)

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      ratio = 1.0
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param
        w_norm = linalg_ops.norm(param, ord=2)
        g_norm = linalg_ops.norm(update, ord=2)
        ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      # update_with_lr = self.learning_rate * update
      # condition = tf.greater(ratio, 1.0)
      # update_with_lr = tf.where(condition, 1.0, ratio) * self.learning_rate
        # * update
      tf.logging.info("*********** I'm using LAMB correction ***********")
      update_with_lr = ratio * self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

BATCH_SIZE = 512 #@param {type:"integer"}
MOMENTUM = 0.9 #@param {type:"number"}
EPOCHS = 30 #@param {type:"integer"}
BUCKET = 'gs://cifar10-data' #@param {type:"string"}
#TPU_ADDRESS = 'infer1'

tf.flags.DEFINE_float("learning_rate", 0.006, "Learning rate.")
tf.flags.DEFINE_float("poly_power", 0.5, "The power of poly decay scheme.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("iterations", 50,
                                "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")
tf.flags.DEFINE_integer("batch_size", 512, "***")
tf.flags.DEFINE_float("weight_decay", 0.01, "***")
tf.flags.DEFINE_integer("warm_up", 3, "***")
tf.flags.DEFINE_string("tpu_name", "infer2", "***")

FLAGS = tf.flags.FLAGS

TPU_ADDRESS = FLAGS.tpu_name
LEARNING_RATE = FLAGS.learning_rate
WARMUP = FLAGS.warm_up
WEIGHT_DECAY = FLAGS.weight_decay


print('Using TPU:', TPU_ADDRESS)
print('learning rate:', LEARNING_RATE)
print('warmup:', WARMUP)
print('weight decay:', WEIGHT_DECAY)

def get_ds_from_tfrec(data_dir, training, batch_size, num_parallel_calls=12, prefetch=8, dtype=tf.float32):

  def _parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features["image"], tf.uint8)
    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])
    image = tf.cast(image, dtype)
    image = (image - [125.30691805, 122.95039414, 113.86538318]) / [62.99321928, 62.08870764, 66.70489964]
    
    label = features["label"]

    if training:
      image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='reflect')
      image = tf.random_crop(image, [32, 32, 3])
      image = tf.image.random_flip_left_right(image)

    return image, label

  split = 'train' if training else 'test'
  filename = os.path.join(data_dir, split + ".tfrecords")
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.repeat()
  dataset = dataset.map(_parser, num_parallel_calls=num_parallel_calls)

  if training:
    dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(prefetch)

  return dataset

#train_input_fn = lambda params: get_ds_from_tfrec(BUCKET + '/cifar-10', training=True, batch_size=params['batch_size'])
#eval_input_fn = lambda params: get_ds_from_tfrec(BUCKET + '/cifar-10', training=False, batch_size=params['batch_size'])
train_input_fn = lambda params: get_ds_from_tfrec(BUCKET, training=True, batch_size=params['batch_size'])
eval_input_fn = lambda params: get_ds_from_tfrec(BUCKET, training=False, batch_size=params['batch_size'])

def init_pytorch(shape, dtype=tf.float32, partition_info=None):
  fan = np.prod(shape[:-1])
  bound = 1 / math.sqrt(fan)
  return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

class ConvBN(tf.keras.Model):
  def __init__(self, c_out):
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

  def call(self, inputs):
    return tf.nn.relu(self.bn(self.conv(inputs)))
  
class Blk(tf.keras.Model):
  def __init__(self, c_out, pool):
    super().__init__()
    self.conv_bn = ConvBN(c_out)
    self.pool = pool

  def call(self, inputs):
    return self.pool(self.conv_bn(inputs))
  
class ResBlk(tf.keras.Model):
  def __init__(self, c_out, pool):
    super().__init__()
    self.blk = Blk(c_out, pool)
    self.res1 = ConvBN(c_out)
    self.res2 = ConvBN(c_out)

  def call(self, inputs):
    h = self.blk(inputs)
    return h + self.res2(self.res1(h))
  
class DavidNet(tf.keras.Model):
  def __init__(self, c=64, weight=0.125):
    super().__init__()
    pool = tf.keras.layers.MaxPooling2D()
    self.init_conv_bn = ConvBN(c)
    self.blk1 = ResBlk(c*2, pool)
    self.blk2 = Blk(c*4, pool)
    self.blk3 = ResBlk(c*8, pool)
    self.pool = tf.keras.layers.GlobalMaxPool2D()
    self.linear = tf.keras.layers.Dense(10, kernel_initializer=init_pytorch, use_bias=False)
    self.weight = weight

  def call(self, x):
    h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
    return self.linear(h) * self.weight
  
  def compute_grads(self, loss):
    var = self.trainable_variables
    grads = tf.gradients(loss, var)
    for g, v in zip(grads, self.trainable_variables):
      g += v * WEIGHT_DECAY * BATCH_SIZE
    return grads

steps_per_epoch = 50000 // BATCH_SIZE

def model_fn(features, labels, mode, params):
  phase = 1 if mode == tf.estimator.ModeKeys.TRAIN else 0
  tf.keras.backend.set_learning_phase(phase)

  model = DavidNet()
  logits = model(features)
  
  #step = tf.train.get_or_create_global_step()
  #lr_schedule = lambda t: tf.cond(tf.less_equal(t, WARMUP), lambda: t * LEARNING_RATE / WARMUP, lambda: (EPOCHS-t) * LEARNING_RATE / (EPOCHS - WARMUP))
  #lr_func = lambda: lr_schedule(tf.cast(step, tf.float32)/steps_per_epoch)/BATCH_SIZE

  #opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)
  #opt = tf.contrib.tpu.CrossShardOptimizer(opt, reduction=tf.losses.Reduction.SUM)
  num_warmup_steps = WARMUP * 50000 // BATCH_SIZE
  num_train_steps = EPOCHS * 50000 // BATCH_SIZE

  loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, reduction=tf.losses.Reduction.SUM)

  #grads = model.compute_grads(loss)
  with tf.control_dependencies(model.get_updates_for(features)):
    #train_op = opt.apply_gradients(zip(grads, model.trainable_variables), global_step=step)
    train_op = create_optimizer(loss, LEARNING_RATE, num_train_steps, num_warmup_steps, True,
                     1.0, 0, WEIGHT_DECAY)

  classes = tf.math.argmax(logits, axis=-1)
  metric_fn = lambda classes, labels: {'accuracy': tf.metrics.accuracy(classes, labels)}
  tpu_metrics = (metric_fn, [classes, labels])
  
  return tf.contrib.tpu.TPUEstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metrics = tpu_metrics
  )

now = datetime.datetime.now()
MODEL_DIR = BUCKET+"/cifar10jobs/job" + "-{}-{:02d}-{:02d}-{:02d}:{:02d}:{:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)

training_config = tf.contrib.tpu.RunConfig(
    cluster=tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS),
    model_dir=MODEL_DIR,
    tpu_config=tf.contrib.tpu.TPUConfig(
    iterations_per_loop=steps_per_epoch,
    per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))
   
estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=10000,
    config=training_config)

print("*********** I have started the training ***********")
estimator.train(train_input_fn, steps=steps_per_epoch*EPOCHS)
print("*********** I have finished the training ***********")

eval_results = estimator.evaluate(input_fn=eval_input_fn, steps=2)
print('\nEvaluation results:\n\t%s\n' % eval_results)
