from audioop import cross
from cProfile import label

from numpy.ma import masked
import tensorflow as tf
import numpy as np

from baseModel import BaseModel
from utils.bidirectional_encoder import BidirectionalLSTMEncoder

class CaptionGenerator(BaseModel):

    def build(self):
        """ Build the model. """
        self.build_cnn()
        if self.config.open_bidirection == True:
            self.build_rnn_bidirectional_atten()
        elif self.config.open_atten:
            self.build_rnn_atten()
        else:
            self.build_rnn()
        
        if self.is_train:
            self.build_optimizer()
            self.build_summary()

    def test_cnn(self,image):
        self.imgs = image
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        self.build_vgg16(images)
        self.probs = tf.nn.softmax(self.fc3l)

        return self.probs

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN...")

        config = self.config
        images = tf.placeholder( dtype=tf.float32, shape=[config.batch_size] + self.image_shape)
        if self.config.cnn == 'vgg16':
            self.build_vgg16(images)

        print("CNN built.")

    def build_vgg16(self,images):
        """ Build the VGG16 net. """
        config = self.config
        # conv1_1
        with tf.variable_scope('conv1_1',reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(name='conv1_1_W',initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1),trainable=config.trainable_variable)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv1_1_b',initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=config.trainable_variable)
            out1_1 = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out1_1)


        # # conv1_2
        with tf.variable_scope('conv1_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='conv1_2_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=config.trainable_variable, name='conv1_2_b')
            out1_2 = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out1_2)
        #
        #
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')
        #
        # # conv2_1
        with tf.variable_scope('conv2_1', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='conv2_1_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv2_1_b')
            out2_1 = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out2_1)
        #
        #
        # # conv2_2
        with tf.variable_scope('conv2_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='conv2_2_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv2_2_b')
            out2_2 = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out2_2)
        #
        #
        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')
        # # conv3_1
        with tf.variable_scope('conv3_1', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='conv3_1_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv3_1_b')
            out3_1 = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out3_1)
        #
        #
        # # conv3_2
        with tf.variable_scope('conv3_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='conv3_2_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv3_2_b')
            out3_2 = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out3_2)
        #
        #
        # # conv3_3
        with tf.variable_scope('conv3_3', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='conv3_3_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv3_3_b')
            out3_3 = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out3_3)

        # # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')
        #
        # # conv4_1
        with tf.variable_scope('conv4_1', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='conv4_1_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv4_1_b')
            out4_1 = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out4_1)
        #
        #
        # # conv4_2
        with tf.variable_scope('conv4_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='conv4_2_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv4_2_b')
            out4_2 = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out4_2)
        #
        #
        # # conv4_3
        with tf.variable_scope('conv4_3', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='conv4_3_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv4_3_b')
            out4_3 = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out4_3)
        #
        #
        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')
        #
        # # conv5_1
        with tf.variable_scope('conv5_1', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='conv5_1_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv5_1_b')
            out5_1 = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out5_1)
        #
        #
        # # conv5_2
        with tf.variable_scope('conv5_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='conv5_2_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv5_2_b')
            out5_2 = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out5_2)
        #
        #
        # # conv5_3
        #
        with tf.variable_scope('conv5_3', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='conv5_3_W',trainable=config.trainable_variable)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=config.trainable_variable, name='conv5_3_b')
            out5_3 = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out5_3)
        #
        #
        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')
        # # fc1
        with tf.variable_scope('fc6', reuse=tf.AUTO_REUSE) as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.get_variable(initializer=tf.truncated_normal([shape, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='fc6_W',trainable=config.trainable_variable)
            fc1b = tf.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                name='fc6_b',trainable=config.trainable_variable)
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
        #
        # # fc2
        with tf.variable_scope('fc7', reuse=tf.AUTO_REUSE) as scope:
            fc2w = tf.get_variable(initializer=tf.truncated_normal([4096, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='fc7_W',trainable=config.trainable_variable)
            fc2b = tf.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=config.trainable_variable, name='fc7_b')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
        #
        # fc3
        with tf.variable_scope('fc8',reuse=tf.AUTO_REUSE) as scope:
            fc3w = tf.get_variable(initializer=tf.truncated_normal([4096, 1000],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='fc8_W',trainable=config.trainable_variable)
            fc3b = tf.get_variable(initializer=tf.constant(1.0, shape=[1000], dtype=tf.float32),
                               trainable=True, name='fc8_b')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)

        # self.conv_feats = self.fc2
        # shape: [batch, 4096]
        ## Reshaping the 4096 to fit the lstm size
        reshaped_fc2_feats = tf.reshape(self.fc2,
                                            [config.batch_size, 8, 512]) # # shape: [batch, 8, 512]
        self.num_ctx = 8
        self.dim_ctx = 512
        self.images = images
        self.conv_feats = reshaped_fc2_feats


    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        # Setup the placeholders
        if self.is_train:
            #contexts = self.conv_feats
            sentences = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size, config.max_caption_length])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.max_caption_length])


        # Setup the word embedding, we can use pre trained word embeddings later //TODO
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Setup the LSTM
        """
        #LSTMCELL tf 1 version - https://github.com/agrawalnishant/tensorflow-1/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard5/tf.nn.rnn_cell.LSTMCell.md
        """
        #tf.nn.rnn_cell.LSTMCell.__init__(num_units, input_size=None, use_peepholes=False, cell_clip=None, initializer=None, num_proj=None, num_unit_shards=1, num_proj_shards=1, forget_bias=1.0, state_is_tuple=False, activation=tanh)
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            # Dropout for training - https://stackoverflow.com/questions/45507315/what-exactly-does-tf-contrib-rnn-dropoutwrapper-in-tensorflow-do-three-cit
            # Source - https://stackoverflow.com/q/45507315 
            # __init__(cell, input_keep_prob=1.0, output_keep_prob=1.0, state_keep_prob=1.0, variational_recurrent=False, input_size=None, dtype=None, seed=None)
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)

        #self.num_ctx = 8
        #self.dim_ctx = 512
        ## 8 * 512 is reduced to 512(embedding size of word) by mean, We can use matrix multiplication also to covert // TODO
        initial_memory = tf.zeros([config.batch_size, lstm.state_size[0]])
        initial_output = tf.zeros([config.batch_size, lstm.state_size[1]])

        # Prepare to run
        predictionsArr = []
        cross_entropies = []
        predictions_correct = []
        num_steps = config.max_caption_length
        image_emb = tf.reduce_mean(self.conv_feats, axis=1)  #shape: [batch, 512] whiwch is same as 

        ## Initial memory and output are given zeros
        last_memory = initial_memory
        last_output = initial_output
        last_word = image_emb


        last_state = last_memory, last_output

        # Generate the words one by one
        for idx in range(num_steps):
            # Embed the last word
            ## for 1st LSTM the input is the image
            if idx == 0:
                word_embed = image_emb
            else:
                with tf.variable_scope("word_embedding"):
                    word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                        last_word)
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = word_embed
                output, state = lstm(current_input, last_state)
                memory, _ = state

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = output
                ## Logits is of size vocab
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                ## Prediction is the index of the word the predicted in the vocab
                prediction = tf.argmax(logits, 1)
                predictionsArr.append(prediction)

                self.probs = probs

            if self.is_train:
                # Compute the loss for this step, if necessary
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = sentences[:, idx],
                    logits = logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)


            last_state = state
            if self.is_train:
                ##  During training the input to LSTM is fed by user
                last_word = sentences[:, idx]
            else:
                # During testing the input to current time stamp of LSTM is the previous time stamp output.
                last_word = prediction

            tf.get_variable_scope().reuse_variables()
        if self.is_train:
            # Compute the final loss, if necessary
            cross_entropies = tf.stack(cross_entropies, axis=1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)

            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + reg_loss

            predictions_correct = tf.stack(predictions_correct, axis=1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)

            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
        self.predictions = tf.stack(predictionsArr, axis=1)

        print("RNN built.")

    # Author: Jianfeng Chen
    # UNI: jc6715
    # Date: 12/09/2025
    # Course: EECS E4040 - Deep Learning & Neural Networks
    # Assignment: Group Project

    # Note: I wrote this code myself, except where I have clearly mentioned references or collaborations.

    # If I collaborated or referred to external sources, I have listed them below:
    # References / Collaborations:
    # 1. Gain the ideas from adding atten machnisim into our original repo, but not just copy, we change the implementation logots from torch to tf1 version- show, attend and tell - https://github.com/AaronCCWong/Show-Attend-and-Tell/tree/f1ac2709224d1a30cd911d6fee8ca2f16886254b
    # 2. TensorFlow 1.x tf.compat.v1.variable_scope documentation - https://www.tensorflow.org/api_docs/python/tf/compat/v1/variable_scope
    # 3. Learn the atten-lstm theroitical knowledge from https://zhuanlan.zhihu.com/p/618223499 with Context, alignment scores, query and value, etc.
    def build_rnn_atten(self):
        """ Build the RNN with attention mechanism. """
        print("Building the RNN with attention...")
        config = self.config
        contexts = self.conv_feats
        # Setup the placeholders
        if self.is_train:
            sentences = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size, config.max_caption_length])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.max_caption_length])


        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)

        # Initialize the LSTM state - wrote on 12/08/2025
        # contexts_mean = tf.reduce_mean(contexts, axis=1)
        # initial_memory, initial_output = self.initialize(contexts_mean)

        #with tf.variable_scope("initialize"): # 12/09/2025 optimization - the 1st time step no need to attend, use mean context as the follwing attend block does
        contexts_mean = tf.reduce_mean(contexts, axis=1)
        initial_memory, initial_output = self.initialize(contexts_mean)

        # Prepare to run
        predictionsArr = []
        cross_entropies = []
        predictions_correct = []
        num_steps = config.max_caption_length

        last_memory = initial_memory
        last_output = initial_output
        last_state = last_memory, last_output
        last_word_input = contexts_mean


        # for idx in range(num_steps):
        #     with tf.variable_scope("attend"):
        #         alpha = self.attend(contexts, last_output)
        #         weighted_contexts = tf.reduce_sum(contexts * tf.expand_dims(alpha, 2), axis=1) #2025/12/08

        for idx in range(num_steps):
            # with tf.variable_scope("attend"):
            if idx > 0:
                alpha = self.attend(contexts, last_output)
                weighted_contexts = tf.reduce_sum(contexts * tf.expand_dims(alpha, 2), axis=1)
            else:
                weighted_contexts = tf.reduce_mean(contexts, axis=1) #2025/12/09 optimization: the 1st time step no need to attend, use mean context


            with tf.variable_scope("word_embedding"):
                if idx == 0:
                    word_embed = last_word_input
                else:
                    word_embed = tf.nn.embedding_lookup(embedding_matrix, last_word_input)

            with tf.variable_scope("lstm"):
                current_input = tf.concat([weighted_contexts, word_embed], axis=1) # [batch, dim_ctx + dim_embed]
                # weighted_contexts: [batch, 512]
                # word_embed: [batch, 512]
                # current_input: [batch, 1024]
                output, state = lstm(current_input, last_state)
                memory, _ = state
            
            # decode and caclulate loss 
            with tf.variable_scope("decode"):
                expanded_output = output
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictionsArr.append(prediction)

                self.probs = probs
            
            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = sentences[:, idx], logits = logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(tf.equal(prediction, ground_truth),
                                              tf.cast(masks[:, idx], tf.float32),
                                              tf.cast(tf.zeros_like(prediction), tf.float32))

                predictions_correct.append(prediction_correct)
            
            if self.is_train:
                last_word_input = sentences[:, idx]
            else:
                last_word_input = prediction

            last_state = state
            last_output = output # !!! 2025/12/09 bug fixed - really important to update output in loop
            tf.get_variable_scope().reuse_variables()
            

        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis=1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                / tf.reduce_sum(masks)

            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + reg_loss

            predictions_correct = tf.stack(predictions_correct, axis=1)
            accuracy = tf.reduce_sum(predictions_correct) \
                    / tf.reduce_sum(masks)

            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
        self.predictions = tf.stack(predictionsArr, axis=1)

        print("RNN with Attention built.")

    # uncomment this following code bolck from the repo and add comments - 12/08/2025
    # add the tf.variable_scope in the inner level for initialize function - 12/09/2025
    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config
        with tf.variable_scope("initialize", reuse=tf.AUTO_REUSE):
            context_mean = self.nn.dropout(context_mean)
            if config.num_initalize_layers == 1:
                # use 1 fc layer to initialize
                memory = self.nn.dense(context_mean,
                                        units = config.num_lstm_units,
                                        activation = None,
                                        name = 'fc_a')
                output = self.nn.dense(context_mean,
                                        units = config.num_lstm_units,
                                        activation = None,
                                        name = 'fc_b')
            else:
                # use 2 fc layers to initialize
                temp1 = self.nn.dense(context_mean,
                                        units = config.dim_initalize_layer,
                                        activation = tf.tanh,
                                        name = 'fc_a1')
                temp1 = self.nn.dropout(temp1)
                memory = self.nn.dense(temp1,
                                        units = config.num_lstm_units,
                                        activation = None,
                                        name = 'fc_a2')

                temp2 = self.nn.dense(context_mean,
                                        units = config.dim_initalize_layer,
                                        activation = tf.tanh,
                                        name = 'fc_b1')
                temp2 = self.nn.dropout(temp2)
                output = self.nn.dense(temp2,
                                        units = config.num_lstm_units,
                                        activation = None,
                                        name = 'fc_b2')
            return memory, output

    # uncomment this following code bolck from the repo and add comments - 12/08/2025
    # add the tf.variable_scope in the inner level for attend function - 12/09/2025
    def attend(self, contexts, output):
        """ Attention Mechanism. """
        config = self.config
        with tf.variable_scope("attend", reuse=tf.AUTO_REUSE):
            reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
            reshaped_contexts = self.nn.dropout(reshaped_contexts)
            output = self.nn.dropout(output)
            if config.num_attend_layers == 1:
                # use 1 fc layer to attend
                logits1 = self.nn.dense(reshaped_contexts,
                                        units = 1,
                                        activation = None,
                                        use_bias = False,
                                        name = 'fc_a')
                logits1 = tf.reshape(logits1, [-1, self.num_ctx])
                logits2 = self.nn.dense(output,
                                        units = self.num_ctx,
                                        activation = None,
                                        use_bias = False,
                                        name = 'fc_b')
                logits = logits1 + logits2
            else:
                # use 2 fc layers to attend
                temp1 = self.nn.dense(reshaped_contexts,
                                        units = config.dim_attend_layer,
                                        activation = tf.tanh,
                                        name = 'fc_1a')
                temp2 = self.nn.dense(output,
                                        units = config.dim_attend_layer,
                                        activation = tf.tanh,
                                        name = 'fc_1b')
                temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
                temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
                temp = temp1 + temp2
                temp = self.nn.dropout(temp)
                logits = self.nn.dense(temp,
                                        units = 1,
                                        activation = None,
                                        use_bias = False,
                                        name = 'fc_2')
                logits = tf.reshape(logits, [-1, self.num_ctx])
            alpha = tf.nn.softmax(logits)
            return alpha

    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        # with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
        #     expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    # """
    # # Author: [Jianfeng Chen] 
    # # UNI: [jc6175]
    # # Date: [2025/12/13]
    # # Course: EECS E4040 - Deep Learning & Neural Networks
    # # Assignment: Group Project

    # # Note: I wrote this code myself, except where I have clearly mentioned references or collaborations.

    # # If I collaborated or referred to external sources, I have listed them below:
    # # References / Collaborations:
    # - just based on the codes of build rnn_atten I wrote to do some modifications 
    # - but the core codes of build function of BILSTM is based on the same references from bidirectional_encoder.py
    # """
    def build_rnn_bidirectional_atten(self):
        """ 
        Build the RNN with bidirectional LSTM and attention mechanism.
        """
        print("Building the RNN with Bidirectional LSTM and Attention...")
        config = self.config
        contexts = self.conv_feats  
        
        # Setup the placeholders
        if self.is_train:
            sentences = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size, config.max_caption_length])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.max_caption_length])

        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Initialize decoder LSTM state in  slef-designed LSTM
        contexts_mean = tf.reduce_mean(contexts, axis=1)  
        
        if self.is_train:
            with tf.variable_scope("bidirectional_caption_encoder"):
                embedded_sentences = tf.nn.embedding_lookup(
                    embedding_matrix, sentences)  # shape is [batch, max_length, dim_embed]
                
                bidirectional_encoder = BidirectionalLSTMEncoder(
                    num_units=config.num_lstm_units // 4,  # 2 is vely slow for training so we choose 4
                    initializer=self.nn.fc_kernel_initializer,
                    dropout_rate=config.lstm_drop_rate,
                    is_train=True
                )
                

                sequence_lengths = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)  # [batch]
                _, caption_final_state = bidirectional_encoder.encode(
                    inputs=embedded_sentences,
                    #sequence_length=None 
                    sequence_length=sequence_lengths  # fix bug on 12/13: error if we choose None, in this case we gonna use actual lengths to avoid padding, really important!!
                )
                
                concatenated_memory, concatenated_output = bidirectional_encoder.get_concatenated_final_state(
                    caption_final_state
                )
                # TODO: might lead to some momory loss if we did not use _ but concatenated_memory
                # concatenated_memory shape is  [batch, num_lstm_units]
                # concatenated_output shape is the sane as [batch, num_lstm_units]
                
                combined_context = tf.concat([contexts_mean, concatenated_output], axis=1)

                initial_memory, initial_output = self.bidirectional_initialize_combined(combined_context)
        else:
            initial_memory, initial_output = self.bidirectional_initialize(contexts_mean)

        # with tf.variable_scope("decoder_lstm"): this was commented as the scope we will use in the inner for-loop
        decoder_lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer=self.nn.fc_kernel_initializer)
        if self.is_train:
            decoder_lstm = tf.nn.rnn_cell.DropoutWrapper(
                decoder_lstm,
                input_keep_prob=1.0-config.lstm_drop_rate,
                output_keep_prob=1.0-config.lstm_drop_rate,
                state_keep_prob=1.0-config.lstm_drop_rate)

        # contexts: [batch, num_ctx, dim_ctx] - might be some spatial features, not setences 

        # Prepare to run
        predictionsArr = []
        cross_entropies = []
        predictions_correct = []
        num_steps = config.max_caption_length

        last_memory = initial_memory
        last_output = initial_output
        last_state = last_memory, last_output
        last_word_input = contexts_mean

        for idx in range(num_steps):
            # Attention mechanism using CNN features directly
            # contexts are spatial features, not sequences, so we use them directly
            if idx > 0:
                alpha = self.bidirectional_attend(contexts, last_output)
                weighted_contexts = tf.reduce_sum(
                    contexts * tf.expand_dims(alpha, 2), axis=1)
            else:
                weighted_contexts = contexts_mean

            with tf.variable_scope("word_embedding", reuse=tf.AUTO_REUSE):
                if idx == 0:
                    word_embed = last_word_input
                else:
                    word_embed = tf.nn.embedding_lookup(embedding_matrix, last_word_input)

            with tf.variable_scope("decoder_lstm", reuse=tf.AUTO_REUSE):
                current_input = tf.concat([weighted_contexts, word_embed], axis=1)
                # current_input: [batch, dim_ctx + dim_embed] = [batch, 1024]
                output, state = decoder_lstm(current_input, last_state)
                memory, _ = state

            # Decode and calculate loss
            with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
                expanded_output = output
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictionsArr.append(prediction)

                self.probs = probs

            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=sentences[:, idx], logits=logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

            if self.is_train:
                last_word_input = sentences[:, idx]
            else:
                last_word_input = prediction

            last_state = state
            last_output = output
            #tf.get_variable_scope().reuse_variables()

        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis=1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                / tf.reduce_sum(masks)

            reg_loss = tf.losses.get_regularization_loss()
            total_loss = cross_entropy_loss + reg_loss

            predictions_correct = tf.stack(predictions_correct, axis=1)
            accuracy = tf.reduce_sum(predictions_correct) \
                    / tf.reduce_sum(masks)

            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
        self.predictions = tf.stack(predictionsArr, axis=1)

        print("RNN with Bidirectional LSTM and Attention built.")

    def bidirectional_attend(self, contexts, output):
        """ Bidirectional LSTM Attention Mechanism. """
        config = self.config
        with tf.variable_scope("bidirectional_attend", reuse=tf.AUTO_REUSE):
            reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
            reshaped_contexts = self.nn.dropout(reshaped_contexts)
            output = self.nn.dropout(output)
            if config.num_attend_layers == 1:
                # use 1 fc layer to attend
                logits1 = self.nn.dense(reshaped_contexts,
                                        units=1,
                                        activation=None,
                                        use_bias=False,
                                        name='fc_a')
                logits1 = tf.reshape(logits1, [-1, self.num_ctx])
                logits2 = self.nn.dense(output,
                                        units=self.num_ctx,
                                        activation=None,
                                        use_bias=False,
                                        name='fc_b')
                logits = logits1 + logits2
            else:
                # use 2 fc layers to attend
                temp1 = self.nn.dense(reshaped_contexts,
                                        units=config.dim_attend_layer,
                                        activation=tf.tanh,
                                        name='fc_1a')
                temp2 = self.nn.dense(output,
                                        units=config.dim_attend_layer,
                                        activation=tf.tanh,
                                        name='fc_1b')
                temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
                temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
                temp = temp1 + temp2
                temp = self.nn.dropout(temp)
                logits = self.nn.dense(temp,
                                        units=1,
                                        activation=None,
                                        use_bias=False,
                                        name='fc_2')
                logits = tf.reshape(logits, [-1, self.num_ctx])
            alpha = tf.nn.softmax(logits)
            return alpha

    def bidirectional_initialize(self, context_mean):
        """ 
        Initialize the decoder LSTM using image context.
        """
        config = self.config
        with tf.variable_scope("bidirectional_initialize", reuse=tf.AUTO_REUSE):
            context_mean = self.nn.dropout(context_mean)
            if config.num_initalize_layers == 1:
                # use 1 fc layer to initialize
                memory = self.nn.dense(context_mean,
                                        units=config.num_lstm_units,
                                        activation=None,
                                        name='fc_a')
                output = self.nn.dense(context_mean,
                                        units=config.num_lstm_units,
                                        activation=None,
                                        name='fc_b')
            else:
                # use 2 fc layers to initialize
                temp1 = self.nn.dense(context_mean,
                                        units=config.dim_initalize_layer,
                                        activation=tf.tanh,
                                        name='fc_a1')
                temp1 = self.nn.dropout(temp1)
                memory = self.nn.dense(temp1,
                                        units=config.num_lstm_units,
                                        activation=None,
                                        name='fc_a2')

                temp2 = self.nn.dense(context_mean,
                                        units=config.dim_initalize_layer,
                                        activation=tf.tanh,
                                        name='fc_b1')
                temp2 = self.nn.dropout(temp2)
                output = self.nn.dense(temp2,
                                        units=config.num_lstm_units,
                                        activation=None,
                                        name='fc_b2')
            return memory, output

    def bidirectional_initialize_combined(self, combined_context):
        """ 
        Initialize the decoder LSTM using combined image and caption context.
        Used during training when ground truth captions are available.
        Input: combined_context [batch, dim_ctx + num_lstm_units] = [batch, 1024]
        """
        config = self.config
        with tf.variable_scope("bidirectional_initialize_combined", reuse=tf.AUTO_REUSE):
            combined_context = self.nn.dropout(combined_context)
            if config.num_initalize_layers == 1:
                # use 1 fc layer to initialize
                memory = self.nn.dense(combined_context,
                                        units=config.num_lstm_units,
                                        activation=None,
                                        name='fc_a')
                output = self.nn.dense(combined_context,
                                        units=config.num_lstm_units,
                                        activation=None,
                                        name='fc_b')
            else:
                # use 2 fc layers to initialize
                temp1 = self.nn.dense(combined_context,
                                        units=config.dim_initalize_layer,
                                        activation=tf.tanh,
                                        name='fc_a1')
                temp1 = self.nn.dropout(temp1)
                memory = self.nn.dense(temp1,
                                        units=config.num_lstm_units,
                                        activation=None,
                                        name='fc_a2')

                temp2 = self.nn.dense(combined_context,
                                        units=config.dim_initalize_layer,
                                        activation=tf.tanh,
                                        name='fc_b1')
                temp2 = self.nn.dropout(temp2)
                output = self.nn.dense(temp2,
                                        units=config.num_lstm_units,
                                        activation=None,
                                        name='fc_b2')
            return memory, output

    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)


        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
