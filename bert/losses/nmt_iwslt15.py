import tensorflow as tf
from losses.loss_base import BaseLoss
from models.nmt import attention_model
from train_flags import FLAGS


class nmt_iwslt15(BaseLoss):

  def loss_fn(self):
    with tf.device('/cpu:0'):
      next_batch = self._dataset_iterator.get_next()

    train_model = attention_model.AttentionModel(
      self._hparams,
      src_ids=next_batch['src_ids'],
      src_seq_len=next_batch['src_len'],
      tgt_ids=next_batch['tgt_ids'],
      tgt_seq_len=next_batch['tgt_len'],
      tgt_ids_out=next_batch['tgt_ids_out'],
      batch_size=FLAGS.batch_size,
      mode=tf.contrib.learn.ModeKeys.TRAIN,
      scope=None,
      extra_args=None)

    self._loss = train_model.train_loss
    return self._loss