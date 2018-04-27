import tensorflow as tf

class MyOptimizer(tf.train.AdamOptimizer):
  def __init__(self, lr):
    super(MyOptimizer, self).__init__(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adam")

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    lr = self._lr_t * tf.sqrt(1 - self._beta2_t) / (1 - self._beta1_t)

    tf.summary.scalar('learning_rate', lr)

    return super(MyOptimizer, self)._apply_dense(grad, var)