import _pickle
import argparse
import os

import numpy as np
import tensorflow as tf
from model import *


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save', help='model directory to store checkpoints')
    parser.add_argument('--input', type=float, default='0.5', help='input value')

    args = parser.parse_args()
    predict(args)


def predict(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = _pickle.load(f)
        model = Model(saved_args, training=False)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                x = args.input
                y = np.sin(x)
                [y_] = model.predict(sess, [x])
                print("actual = ", y, " predicted = ", y_)


if __name__ == '__main__':
    main()
