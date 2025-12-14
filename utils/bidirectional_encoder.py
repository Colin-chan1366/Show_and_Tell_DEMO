"""
Rebuild the Bidirectional LSTM Encoder Class Implementation for TensorFlow 1.x
Based on our Assignment 3 TF2/Keras version to write the TF1 compatible version
# Author: [Jianfeng Chen] 
# UNI: [jc6175]
# Date: [2025/12/13]
# Course: EECS E4040 - Deep Learning & Neural Networks
# Assignment: Group Project

# Note: I wrote this code myself, except where I have clearly mentioned references or collaborations.

# If I collaborated or referred to external sources, I have listed them below:
# References / Collaborations:
# - learn the princile of bidirectional LSTM 1- https://medium.com/@anishnama20/understanding-bidirectional-lstm-for-sequential-data-processing-b83d6283befc
# - learn the prinicle of bidirectional LSTM 2 - https://zhuanlan.zhihu.com/p/40119926
# - rebuild the calss of bidirection lstm based on Assignment 3 task3, last bonus part as well as utils/translation
/layers.py the build function of tf.keras.layers.Bidirectional - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional 
# and tf1 keras LSTM cell doc - https://github.com/agrawalnishant/tensorflow-1/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard5/tf.nn.rnn_cell.LSTMCell.md
# - based on Assignment 3 /utils/LSTM.py for build up the tf.nn.rnn_cell.LSTMCell
# - tf1 dynamic bidirectional rnn doc - https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/bidirectional_dynamic_rnn
"""

import tensorflow as tf
import numpy as np


class BidirectionalLSTMEncoder(object):
    def __init__(self, num_units, initializer=None, dropout_rate=0.0, is_train=True):
        self.num_units = num_units
        self.initializer = initializer
        self.dropout_rate = dropout_rate
        self.is_train = is_train
        
    # do not like the  LSTMCell call function using f, i, c, o caculation
    # we used the tf1 implemented tf.nn.rnn_cell.LSTMCell
    def _create_lstm_cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(
            self.num_units,
            initializer=self.initializer
        )
        
        if self.is_train and self.dropout_rate > 0:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                input_keep_prob=1.0 - self.dropout_rate,
                output_keep_prob=1.0 - self.dropout_rate,
                state_keep_prob=1.0 - self.dropout_rate
            )
        
        return cell
    
    # the most important part of experiment 5
    # both update captiona & sentences no matter forward or bacward 
    # 12/12/2025 wrote 
    def encode(self, inputs, sequence_length=None, initial_state_fw=None, initial_state_bw=None, scope=None):

        
        with tf.variable_scope(scope or "bidirectional_encoder", reuse=tf.AUTO_REUSE):
            
            with tf.variable_scope("forward_cell"):
                forward_cell = self._create_lstm_cell()
            with tf.variable_scope("backward_cell"):
                backward_cell = self._create_lstm_cell()
            
            # https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/bidirectional_dynamic_rnn
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_cell,
                cell_bw=backward_cell,
                inputs=inputs, # remember the dim of inputs: # [batch, max_time, input_dim]
                sequence_length=sequence_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32
            )
            
            # remember that outputs here is a tuple (output_fw, output_bw)
            # we have to concat forward and backward output for caculation, unless will lead to a error of dim. Please note here is axis=2 not 1.
            combined_outputs = tf.concat(outputs, axis=2)
            
            return combined_outputs, final_state
    
    def get_combined_final_state(self, final_state):

        state_fw, state_bw = final_state
        memory_fw, output_fw = state_fw
        memory_bw, output_bw = state_bw
        
        combined_memory = memory_fw + memory_bw
        combined_output = output_fw + output_bw
        
        return combined_memory, combined_output
    
    def get_concatenated_final_state(self, final_state):

        state_fw, state_bw = final_state
        memory_fw, output_fw = state_fw
        memory_bw, output_bw = state_bw
        
        concatenated_memory = tf.concat([memory_fw, memory_bw], axis=1)
        concatenated_output = tf.concat([output_fw, output_bw], axis=1)
        
        return concatenated_memory, concatenated_output
