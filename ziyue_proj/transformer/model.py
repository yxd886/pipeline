# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp


    def encode(self, x, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        scopes = []
        outputs = []
        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            self.token2idx, self.idx2token = load_vocab(self.hp.vocab)
            self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)
            scopes.append(tf.get_variable_scope().name)
            outputs.append(self.embeddings)
        with tf.variable_scope("encoder_embedding_lookup", reuse=tf.AUTO_REUSE):
            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)
            scopes.append(tf.get_variable_scope().name)
            outputs.append(enc)
            ## Blocks
        for i in range(self.hp.num_blocks):
            with tf.variable_scope("encoder_num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                enc = multihead_attention(queries=enc,
                                          keys=enc,
                                          values=enc,
                                          key_masks=src_masks,
                                          num_heads=self.hp.num_heads,
                                          dropout_rate=self.hp.dropout_rate,
                                          training=training,
                                          causality=False)
                # feed forward
                enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
                scopes.append(tf.get_variable_scope().name)
                outputs.append(enc)
        memory = enc
        return memory, src_masks,outputs,scopes

    def decode(self, decoder_inputs, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        scopes = []
        outputs = []
        with tf.variable_scope("decoder_embedding_lookup", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys
            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)
            scopes.append(tf.get_variable_scope().name)
            outputs.append(dec)
            # Blocks
        for i in range(self.hp.num_blocks):
            with tf.variable_scope("decoder_num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # Masked self-attention (Note that causality is True at this time)
                dec = multihead_attention(queries=dec,
                                          keys=dec,
                                          values=dec,
                                          key_masks=tgt_masks,
                                          num_heads=self.hp.num_heads,
                                          dropout_rate=self.hp.dropout_rate,
                                          training=training,
                                          causality=True,
                                          scope="self_attention")

                # Vanilla attention
                dec = multihead_attention(queries=dec,
                                          keys=memory,
                                          values=memory,
                                          key_masks=src_masks,
                                          num_heads=self.hp.num_heads,
                                          dropout_rate=self.hp.dropout_rate,
                                          training=training,
                                          causality=False,
                                          scope="vanilla_attention")
                ### Feed Forward
                dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])
                scopes.append(tf.get_variable_scope().name)
                outputs.append(dec)


        return dec, sents2,outputs,scopes

    def train(self, xs, decode_inputs,y):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, src_masks,outputs,scopes = self.encode(xs)
        dec,sents2,outputs1,scopes1 = self.decode(decode_inputs, memory, src_masks)
        # Final linear projection (embedding weights are shared)

        outputs = outputs+outputs1
        scopes = scopes+scopes1
        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
            weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
            logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
            y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
            #nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
            #loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
            scopes.append(tf.get_variable_scope().name)
            outputs.append(loss)


        return loss,outputs,scopes


