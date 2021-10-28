import os

import sys

sys.path.append("../")

import nn
from argparse import ArgumentParser

import tensorflow as tf


def main(args):
    # load data
    train_ds, train_size, val_ds, val_size, test_ds, test_size = nn.datasets.cosserat.cosserat_rods_sim_pc(
        path=args.dataset_path,
        content_file=args.content_file,
        batch_size=args.batch_size
    )

    # shape predictor (input dim == output dim)
    num_pts = 26
    seq_length = 19
    cosserat_net = nn.models.cosserat.CosseratNet(args.batch_size, num_pts, seq_length,
                                                  momentum=args.momentum, activation=args.activation)
    cosserat_net.warmup()

    # setup optimization
    lr_f = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, args.lr_decay_steps, args.lr_beta)
    wd_f = tf.keras.experimental.CosineDecay(args.weight_decay, args.weight_decay_steps, args.weight_decay_alpha)

    lr = tf.Variable(args.lr)
    wd = tf.Variable(args.weight_decay)

    optimizer = tf.keras.optimizers.Adam(args.weight_decay, args.lr)

    # experiment handler
    experiment_handler = nn.train.log.ExperimentHandler(
        args.working_path, args.out_name,
        cosserat_net=cosserat_net,
        optimizer=optimizer
    )

    # restore if provided
    if args.restore_path is not None:
        nn.train.log.restore_from_checkpoint_latest(
            path=args.restore_path,
            cosserat_net=cosserat_net,
            optimizer=optimizer
        )

    # @tf.function(input_signature=[
    #     tf.TensorSpec([None, seq_length, 3, num_pts], tf.float32),
    #     tf.TensorSpec([None, 4], tf.float32),
    #     tf.TensorSpec([None, seq_length, 3, num_pts], tf.float32),
    #     tf.TensorSpec([], tf.bool)
    # ])
    def query(x, params, y, training):
        x, x_target, x_params, y = nn.losses.point.prepare_input_rod_data(x, params, y)
        outputs = cosserat_net([x, x_target, x_params], training=training)
        total_loss = []
        loss = nn.losses.point.absoulte(outputs, y)
        total_loss.append(loss)
        return total_loss, outputs

    # @tf.function(input_signature=[
    #     tf.TensorSpec([None, seq_length, 3, num_pts], tf.float32),
    #     tf.TensorSpec([None, 4], tf.float32),
    #     tf.TensorSpec([None, seq_length, 3, num_pts], tf.float32)
    # ])
    def train(x, params, y):
        with tf.GradientTape() as tape:
            model_loss, outputs = query(x, params, y, True)
            total_loss = tf.reduce_mean(model_loss)  # keep unreduced to calculate epoch stats and push reduced to loss

        grads = tape.gradient(total_loss, cosserat_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, cosserat_net.trainable_variables))
        return model_loss, outputs

    # run training and validation
    epoch = 0
    train_step, val_step = 0, 0
    m_model_loss = tf.metrics.Mean('model_loss')

    while True:
        experiment_handler.log_training()

        m_model_loss.reset_states()
        with nn.train.log.as_progressbar('Train', epoch, train_size) as pbar:
            for x, params, y in train_ds:
                lr.assign(lr_f(train_step))
                wd.assign(wd_f(train_step))

                model_loss, pc_pred = train(x, params, y)
                m_model_loss.update_state(model_loss)

                if train_step % args.log_interval == 0:
                    tf.summary.scalar('info/lr', lr, step=train_step)
                    tf.summary.scalar('info/weight_decay', wd, step=train_step)

                    for i, partial_loss in enumerate(model_loss):
                        tf.summary.scalar('metrics/loss_{}'.format(i), tf.reduce_mean(partial_loss), step=train_step)

                train_step += 1
                pbar.update(args.batch_size)

        tf.summary.scalar('epoch/model_loss', m_model_loss.result(), step=epoch)

        experiment_handler.save_last()
        experiment_handler.flush()
        experiment_handler.log_validation()

        m_model_loss.reset_states()

        with nn.train.log.as_progressbar('Val', epoch, val_size) as pbar:
            for x, params, y in val_ds:
                model_loss, pc_pred = query(x, params, y, training=False)
                m_model_loss.update_state(model_loss)

                if val_step % args.log_interval == 0:
                    for i, partial_loss in enumerate(model_loss):
                        tf.summary.scalar('metrics/loss_{}'.format(i), tf.reduce_mean(partial_loss), step=val_step)

                val_step += 1
                pbar.update(args.batch_size)

        tf.summary.scalar('epoch/model_loss', m_model_loss.result(), step=epoch)

        experiment_handler.flush()
        epoch += 1

        if epoch >= args.num_epochs > 0:
            break


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset-path', type=str, default='./')
    parser.add_argument('--content-file', type=str, default='dataset.npy')
    parser.add_argument('--working-path', type=str, default='./workspace')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--log-images-interval-train', type=int, default=100)
    parser.add_argument('--log-images-interval-val', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay-steps', type=int, default=200)
    parser.add_argument('--lr-beta', type=float, default=0.99)
    parser.add_argument('--num-epochs', type=int, default=-1)
    parser.add_argument('--out-name', type=str, default="cosserat")
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--weight-decay-steps', type=float, default=30000)
    parser.add_argument('--weight-decay-alpha', type=float, default=1e-3)

    parser.add_argument('--momentum', type=float, default=0.98)
    parser.add_argument('--activation', type=str, default='relu')

    parser.add_argument('--allow-memory-growth', action='store_true', default=True)
    parser.add_argument('--no-warnings', action='store_true', default=True)
    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        nn.train.device.allow_memory_growth()

    if args.no_warnings:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    main(args)
