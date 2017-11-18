import _pickle
import argparse
import os
import time

import data_loader
import numpy as np
import tensorflow as tf
from model import Model


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/training_data', help='directory to load input data from')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to store tensorboard logs')
    parser.add_argument('--batch_size', type=int, default=50, help='mini batch size')
    parser.add_argument('--num_points', type=int, default=100000, help='number of input points to generate')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000, help='how frequently save checkpoints')
    # optimization hyper parameter. for adam, b1 = 0.9, b2 = 0.999 and lr = 1e-3 or 5*e-4
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    # regularization hyper parameter
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help="probability of keeping weights in the input layer")
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help="probability of keeping weights in the hidden layer")
    # retraining ?
    parser.add_argument('--init_from', type=str, default=None,
                        help="""Continue training from saved model at this path. 
                        Path must contain files saved by previous training session.
                        'config.pkl'    :   configuration;
                        'checkpoint'    :   path to model file(s) created by tf 
                        'model.ckpt=*'  :   file(s) with model definition (created by tf)   
    """)
    args = parser.parse_args()
    train(args)


def train(args):
    dl = data_loader.DataLoader(args)

    if args.init_from is not None:
        assert os.path.isdir(args.init_from), " %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, 'config.pkl')), \
            'config.pkl does not exists in path {}'.format(args.init_from)
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = _pickle.load(f)
        need_to_be_same = ['batch_size']
        for item in need_to_be_same:
            assert vars(saved_model_args)[item] == vars(args)[item], \
                'command line and saved model param not matching for {}'.format(item)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        _pickle.dump(args, f)
    x, y = dl.next_batch()
    model = Model(args)

    with tf.Session() as sess:
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
            os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
        )
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())

        if args.init_from is not None:
            print('model restored.')
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in range(args.num_epochs):
            # update_lr = tf.assign(model.learning_rate, args.learning_rate * (args.deccay_rate ** e))
            # sess.run(update_lr)
            num_batches = args.num_points // args.batch_size
            dl.reset_pointer()
            for b in range(num_batches):
                start = time.time()
                x, y = dl.next_batch()
                x = np.reshape(x, (args.batch_size, 1))
                y = np.reshape(y, (args.batch_size, 1))
                # print("inupt shape", x.shape, y.shape)
                feed = {model.input_data: x, model.targets: y}

                summ, train_loss, _ = sess.run([summaries, model.loss, model.train_step], feed)
                writer.add_summary(summ, e * num_batches + b)
                end = time.time()

                print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'
                      .format(e * num_batches + b, args.num_epochs * num_batches, e, train_loss, end - start))
                if (e * num_batches + b) % args.save_every == 0 \
                        or (e == args.num_epochs - 1 and b == num_batches - 1):
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == "__main__":
    main()
