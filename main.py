#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import numpy as np

"""
https://www.geeksforgeeks.org/python/whats-the-purpose-of-tf-app-flags-in-tensorflow/

import tensorflow as tf
flag = tf.compat.v1.app.flags
FLAGS = flag.FLAGS

flag.DEFINE_integer('n_epochs', 100, 'Number of steps')
flag.DEFINE_float('lr', 0.005, 'Learning rate')
flag.DEFINE_integer('batch_size', 32, 'Batch Size')

print("Default Learning Rate:", FLAGS.lr) # 0.005
print("Default Epochs:", FLAGS.n_epochs) # 100
print("Default Batch Size:", FLAGS.batch_size) # 32
"""
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

tf.flags.DEFINE_string('image_file','./man.jpg','The file to test the CNN')


## Start token is not required, Stop Tokens are given via "." at the end of each sentence.
## TODO : Early stop functionality by considering validation error. We should first split the validation data.

def main(argv):

    # just change the experiment_no when we starts a new one
    config = Config(experiment_name='experiment_5')
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size
    config.trainable_variable = FLAGS.train_cnn

    with tf.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            data = prepare_train_data(config)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            #load the cnn file
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)

        elif FLAGS.phase == 'eval':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)

        elif FLAGS.phase == 'test_loaded_cnn':
            # testing only cnn
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
            probs = model.test_cnn(imgs)
            model.load_cnn(sess, FLAGS.cnn_model_file)

            img1 = imread(FLAGS.image_file, mode='RGB')
            img1 = imresize(img1, (224, 224))

            prob = sess.run(probs, feed_dict={imgs: [img1]})[0]
            preds = (np.argsort(prob)[::-1])[0:5]
            for p in preds:
                print(class_names[p], prob[p])

        else:
            # testing phase
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)

if __name__ == '__main__':
    tf.app.run()