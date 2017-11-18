import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hparams', type=str, default=None, help='comma separated list of "name=value" pair')
    args = parser.parse_args()
    train(args)


def train(args):
    hparams = tf.contrib.training.HParams(learning_rate=0.001, num_hidden_units=100, activations=['relu', 'tanh'])
    hparams.parse(args.hparams)


if __name__ == "__main__":
    main()