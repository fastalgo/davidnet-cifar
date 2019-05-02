from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

import math
import datetime
import os
from typing import List
import fenwicks as fw


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
        ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
          math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      # update_with_lr = self.learning_rate * update
      # condition = tf.greater(ratio, 1.0)
      # update_with_lr = tf.where(condition, 1.0, ratio) * self.learning_rate * update
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


#if tf.gfile.Exists('./fenwicks'):
#  tf.gfile.DeleteRecursively('./fenwicks')
#!git clone https://github.com/fenwickslab/fenwicks.git
#from fenwicks.utils.colab import TPU_ADDRESS
BATCH_SIZE = 512 #@param ["512", "256", "128"] {type:"raw"}
MOMENTUM = 0.9 #@param ["0.9", "0.95", "0.975"] {type:"raw"}
#WEIGHT_DECAY = 0.000125 #@param ["0.000125", "0.00025", "0.0005"] {type:"raw"}
WEIGHT_DECAY = 0.01
WEIGHT_DECAY = 0.064
#LEARNING_RATE = 0.4 #@param ["0.4", "0.2", "0.1"] {type:"raw"}
LEARNING_RATE = 12.00 #@param ["0.4", "0.2", "0.1"] {type:"raw"}
EPOCHS = 24 #@param {type:"slider", min:0, max:100, step:1}
WARMUP = 5 #@param {type:"slider", min:0, max:24, step:1}
BUCKET = 'gs://bert-pretrain-data'
PROJECT = 'cifar'

tf.flags.DEFINE_float("learning_rate", 10.0, "Learning rate.")
tf.flags.DEFINE_float("poly_power", 0.5, "The power of poly decay scheme.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("iterations", 50,
                                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")
tf.flags.DEFINE_integer("batch_size", 512, "***")
tf.flags.DEFINE_float("weight_decay", 0.01, "***")
tf.flags.DEFINE_integer("warm_up", 5, "***")
tf.flags.DEFINE_string("tpu_name", "infer2", "***")

FLAGS = tf.flags.FLAGS

LEARNING_RATE = FLAGS.learning_rate
WARMUP = FLAGS.warm_up
WEIGHT_DECAY = FLAGS.weight_decay
TPU_ADDRESS = FLAGS.tpu_name
#TPU_ADDRESS = 'infer2'


def exp_decay_lr(init_lr: float, decay_steps: int, base_lr: float = 0, decay_rate: float = 1 / math.e):
    """
    Get exponential learning rate decay schedule function.

    Learning rate schedule:

    ```python
    lr = base_lr + init_lr * decay_rate ^ (global_step / decay_steps)
    ```

    :param init_lr: initial learning rate, also the highest value.
    :param decay_steps: number of steps for the learning rate to reduce by a full decay_rate.
    :param base_lr: smallest learning rate. Default: 0.
    :param decay_rate: the decay rate. Default 1/e.
    :return: learning rate schedule function satisfying the above descriptions. The function has one optional parameter:
             the training step count `step`. `step` defaults to `None`, in which case the function gets or creates
             Tensorflow's `global_step`.
    """

    def lr_func(step: tf.Tensor = None) -> tf.Tensor:
        if step is None:
            step = tf.train.get_or_create_global_step()
        return base_lr + tf.train.exponential_decay(init_lr, step, decay_steps, decay_rate)

    return lr_func


def triangular_lr(init_lr: float, total_steps: int, warmup_steps: int):
    """
    One cycle triangular learning rate schedule.

    :param init_lr: peak learning rate.
    :param total_steps: total number of training steps.
    :param warmup_steps: number of steps in the warmup phase, during which the learning rate increases linearly.
    :return: learning rate schedule function satisfying the above descriptions. The function has one optional parameter:
             the training step count `step`. `step` defaults to `None`, in which case the function gets or creates
             Tensorflow's `global_step`.
    """

    def lr_func(step: tf.Tensor = None) -> tf.Tensor:
        if step is None:
            step = tf.train.get_or_create_global_step()
        step = tf.cast(step, tf.float32)
        warmup_sched = lambda: step * init_lr / warmup_steps
        decay_sched = lambda: (total_steps - step) * init_lr / (total_steps - warmup_steps)
        return tf.cond(tf.less_equal(step, warmup_steps), warmup_sched, decay_sched)

    return lr_func


def cosine_lr(init_lr: float, total_steps: int):
    """
    Get Adam optimizer function with one-cycle SGD with Warm Restarts, a.k.a. cosine learning rate decay.

    :param init_lr: initial learning rate, also the highest value.
    :param total_steps: total number of training steps.
    :return: learning rate schedule function satisfying the above descriptions. The function has one optional parameter:
             the training step count `step`. `step` defaults to `None`, in which case the function gets or creates
             Tensorflow's `global_step`.
    """

    def lr_func(step: tf.Tensor = None) -> tf.Tensor:
        if step is None:
            step = tf.train.get_or_create_global_step()
        return tf.train.cosine_decay_restarts(init_lr, step, total_steps)

    return lr_func


def adam_optimizer(lr_func):
    """
    Adam optimizer with a given learning rate schedule.

    :param lr_func: learning rate schedule function.
    :return: optimizer function satisfying the above descriptions.
    """

    def opt_func():
        lr = lr_func()
        return tf.train.AdamOptimizer(lr)

    return opt_func


class SGD(tf.train.MomentumOptimizer):
    def __init__(self, lr: tf.Tensor, mom: float, wd: float):
        super().__init__(lr, momentum=mom, use_nesterov=True)
        self.wd = wd

    def compute_gradients(self, loss, var_list=None):
        grads_and_vars = super().compute_gradients(loss, var_list=var_list)

        l = len(grads_and_vars)
        for i in range(l):
            g, v = grads_and_vars[i]
            g += v * self.wd
            grads_and_vars[i] = (g, v)

        return grads_and_vars


def sgd_optimizer(lr_func, mom: float = 0.9, wd: float = 0.0):
    """
    SGD with Nesterov momentum optimizer with a given learning rate schedule.

    :param lr_func: learning rate schedule function.
    :param mom: momentum for SGD. Default: 0.9
    :param wd: weight decay factor. Default: no weight decay.
    :return: optimizer function satisfying the above descriptions.
    """
    # optimizer = LAMBOptimizer(
    # learning_rate=learning_rate,
    # weight_decay_rate=weight_decay_input,
    # beta_1=0.9,
    # beta_2=0.999,
    # epsilon=1e-6,
    # exclude_from_weight_decay=["LayerNorm", "layer_norm", "batch_normalization", "BatchNormalization", "bias"])
    def opt_func():
        lr = lr_func()
        # return SGD(lr, mom=mom, wd=wd)
        my_opt = LAMBOptimizer(learning_rate=lr, 
          weight_decay_rate=wd,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-6,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "batch_normalization", "BatchNormalization", "bias"])
        return my_opt

    return opt_func


def weight_decay_loss(wd: float = 0.0005) -> tf.Tensor:
    l2_loss = []
    for v in tf.trainable_variables():
        if 'BatchNorm' not in v.name and 'weights' in v.name:
            l2_loss.append(tf.nn.l2_loss(v))
    return wd * tf.add_n(l2_loss)


# fixme
def inception_v3_lr(n_train, lr: float = 0.165, lr_decay: float = 0.94, lr_decay_epochs: int = 3,
                    batch_size: float = 1024, use_warmup: bool = False, warmup_epochs: int = 7, cold_epochs: int = 2):
    init_lr = lr * batch_size / 256

    final_lr = 0.0001 * init_lr

    steps_per_epoch = n_train / batch_size
    global_step = tf.train.get_or_create_global_step()

    current_epoch = tf.cast((tf.cast(global_step, tf.float32) / steps_per_epoch), tf.int32)

    lr = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step,
                                    decay_steps=int(lr_decay_epochs * steps_per_epoch), decay_rate=lr_decay,
                                    staircase=True)

    if use_warmup:
        warmup_decay = lr_decay ** ((warmup_epochs + cold_epochs) / lr_decay_epochs)
        adj_init_lr = init_lr * warmup_decay

        wlr = 0.1 * adj_init_lr
        wlr_height = tf.cast(0.9 * adj_init_lr / (warmup_epochs + lr_decay_epochs - 1), tf.float32)
        epoch_offset = tf.cast(cold_epochs - 1, tf.int32)
        exp_decay_start = (warmup_epochs + cold_epochs + lr_decay_epochs)
        lin_inc_lr = tf.add(wlr, tf.multiply(tf.cast(tf.subtract(current_epoch, epoch_offset), tf.float32), wlr_height))
        lr = tf.where(tf.greater_equal(current_epoch, cold_epochs),
                      (tf.where(tf.greater_equal(current_epoch, exp_decay_start), lr, lin_inc_lr)), wlr)

    lr = tf.maximum(lr, final_lr, name='learning_rate')
    return lr


def get_tpu_estimator(steps_per_epoch: int, model_func, work_dir: str, ws_dir: str = None, ws_vars: List[str] = None,
                      trn_bs: int = 128, val_bs: int = 1, pred_bs: int = 1) -> tf.contrib.tpu.TPUEstimator:
    """
    Create a TPUEstimator object ready for training and evaluation.

    :param steps_per_epoch: Number of training steps for each epoch.
    :param model_func: Model function for TPUEstimator. Can be built with `get_clf_model_func'.
    :param work_dir: Directory for storing intermediate files (such as checkpoints) generated during training.
    :param ws_dir: Directory containing warm start files, usually a pre-trained model checkpoint.
    :param ws_vars: List of warm start variables, usually from a pre-trained model.
    :param trn_bs: Batch size for training.
    :param val_bs: Batch size for validation. Default: all validation records in a single batch.
    :param pred_bs: Batch size for prediction. Default: 1.
    :return: A TPUEstimator object, for training, evaluation and prediction.
    """

    cluster = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)

    tpu_cfg = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=steps_per_epoch,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2)

    now = datetime.datetime.now()
    time_str = '{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}:{now.minute:02d}:{now.second:02d}'
    work_dir = os.path.join(work_dir, time_str)

    trn_cfg = tf.contrib.tpu.RunConfig(cluster=cluster, model_dir=work_dir, tpu_config=tpu_cfg)

    ws = None if ws_dir is None else tf.estimator.WarmStartSettings(ckpt_to_initialize_from=ws_dir,
                                                                    vars_to_warm_start=ws_vars)

    return tf.contrib.tpu.TPUEstimator(model_fn=model_func, model_dir=work_dir, train_batch_size=trn_bs,
                                       eval_batch_size=val_bs, predict_batch_size=pred_bs, config=trn_cfg,
                                       warm_start_from=ws)


def get_clf_model_func(model_arch, opt_func, reduction=tf.losses.Reduction.MEAN):
    """
    Build a model function for a classification task to be used in a TPUEstimator, based on a given model architecture
    and an optimizer. Both the model architecture and optimizer must be callables, not model or optimizer objects. The
    reason for this design is to ensure that all variables are created in the same Tensorflow graph, which is created
    by the TPUEstimator.

    :param model_arch: Model architecture: a callable that builds a neural net model.
    :param opt_func: Optimization function: a callable that returns an optimizer.
    :param reduction: Whether to average (`tf.losses.Reduction.MEAN`) or sum (`tf.losses.Reduction.SUM`) losses
                      for different training records. Default: average.
    :return: Model function ready for TPUEstimator.
    """

    def model_func(features, labels, mode, params):
        phase = 1 if mode == tf.estimator.ModeKeys.TRAIN else 0
        tf.keras.backend.set_learning_phase(phase)

        model = model_arch()
        logits = model(features)
        y_pred = tf.math.argmax(logits, axis=-1)
        train_op = None
        loss = None

        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, reduction=reduction)

        if mode == tf.estimator.ModeKeys.TRAIN:
            step = tf.train.get_or_create_global_step()

            opt = opt_func()
            opt = tf.contrib.tpu.CrossShardOptimizer(opt, reduction=reduction)

            var = model.trainable_variables  # this excludes frozen variables
            grads_and_vars = opt.compute_gradients(loss, var_list=var)
            with tf.control_dependencies(model.get_updates_for(features)):
                train_op = opt.apply_gradients(grads_and_vars, global_step=step)
                new_global_step = step + 1
                train_op = tf.group(train_op, [step.assign(new_global_step)])

        metric_func = lambda y_pred, labels: {'accuracy': tf.metrics.accuracy(y_pred, labels)}
        tpu_metrics = (metric_func, [y_pred, labels])

        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, predictions={"y_pred": y_pred},
                                               train_op=train_op, eval_metrics=tpu_metrics)

    return model_func


data_dir, work_dir = fw.io.get_gcs_dirs(BUCKET, PROJECT)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
n_train, n_test = X_train.shape[0], X_test.shape[0]
img_size = X_train.shape[1]
n_classes = y_train.max() + 1

X_train_mean = np.mean(X_train, axis=(0,1,2))
X_train_std = np.std(X_train, axis=(0,1,2))
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

train_fn = os.path.join(data_dir, "train.tfrec")
test_fn = os.path.join(data_dir, "test.tfrec")

fw.data.numpy_tfrecord(train_fn, X_train, y_train)
fw.data.numpy_tfrecord(test_fn, X_test, y_test)

def parser_train(tfexample):
  x, y = fw.data.tfexample_numpy_image_parser(tfexample, img_size, img_size)
  x = fw.transform.random_pad_crop(x, 4)
  x = fw.transform.random_flip(x)
  x = fw.transform.cutout(x, 8, 8)
  return x, y

parser_test = lambda x: fw.data.tfexample_numpy_image_parser(x, img_size, img_size)

train_input_func = lambda params: fw.data.tfrecord_ds(train_fn, parser_train, batch_size=params['batch_size'], training=True)
#eval_input_func = lambda params: fw.data.tfrecord_ds(test_fn, parser_test, batch_size=params['batch_size'], training=False)
eval_input_func = lambda params: fw.data.tfrecord_ds(test_fn, parser_test, batch_size=1000, training=False)

def build_nn(c=64, weight=0.125):
  model = fw.Sequential()
  model.add(fw.layers.ConvBN(c, **fw.layers.PYTORCH_CONV_PARAMS))
  model.add(fw.layers.ConvResBlk(c*2, res_convs=2, **fw.layers.PYTORCH_CONV_PARAMS))
  model.add(fw.layers.ConvBlk(c*4, **fw.layers.PYTORCH_CONV_PARAMS))
  model.add(fw.layers.ConvResBlk(c*8, res_convs=2, **fw.layers.PYTORCH_CONV_PARAMS))
  model.add(tf.keras.layers.GlobalMaxPool2D())
  model.add(fw.layers.Classifier(n_classes, kernel_initializer=fw.layers.init_pytorch, weight=weight))
  return model

steps_per_epoch = n_train // BATCH_SIZE
total_steps = steps_per_epoch * EPOCHS

lr_func = triangular_lr(LEARNING_RATE/BATCH_SIZE, total_steps, warmup_steps=WARMUP*steps_per_epoch)
#fw.plt.plot_lr_func(lr_func, total_steps)

opt_func = sgd_optimizer(lr_func, mom=MOMENTUM, wd=WEIGHT_DECAY)
model_func = get_clf_model_func(build_nn, opt_func, reduction=tf.losses.Reduction.SUM)
# model_func is a tf.contrib.tpu.TPUEstimatorSpec

est = get_tpu_estimator(steps_per_epoch, model_func, work_dir, trn_bs=BATCH_SIZE, val_bs=n_test)
est.train(train_input_func, steps=total_steps)

result = est.evaluate(eval_input_func, steps=1)

#print('Test results: accuracy={result["accuracy"] * 100: .2f}%, loss={result["loss"]: .2f}.')
#eval_results = estimator.evaluate(input_fn=eval_input_fn, steps=2)
print('\nEvaluation results:\n\t%s\n' % result)

fw.io.create_clean_dir(work_dir)

