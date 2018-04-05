import os
import pdb
import time
import itertools
import numpy as np
import tensorflow as tf

import utils_mnist as umnist
from models_lenet import LeNet
from models_simple import Simple

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

### PARAMS
network = 'simple'
optimizer = 'adam'
batch_size = 1024
num_feats = 10
lr_start = 1e-02
lr_factor = 0.1
lr_strat = [5, 8]
max_epochs = 10
print_epoch = 1
w_decay = 1e-4

exp_name = 'MNIST_LeNet_' + network + '_' + optimizer + '_v0'
path_experiment = 'experiments/' + exp_name
###


def train_net(trn_x, trn_y):
    tf.reset_default_graph()
    # placeholders
    input = tf.placeholder(tf.float32, [batch_size, trn_x.shape[1], trn_x.shape[2], trn_x.shape[3]])
    labels = tf.placeholder(tf.float32, [batch_size, num_feats])
    learning_rate = tf.placeholder(tf.float32, shape=[])

    # build network
    if network == 'simple':
        net = Simple(input)
    elif network == 'lenet':
        net = LeNet(input, num_features=num_feats)

    # add loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net.out, labels=labels))
    loss = loss + w_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    correct_pred = tf.equal(tf.argmax(net.probs, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # choose optimizer
    if optimizer == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=tf.trainable_variables())
    elif optimizer == 'mom':
        train_op = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(loss)

    # train
    with tf.Session(config=config) as sess:
        loss_batch = []
        acc_batch = []
        # initialize graph
        sess.run(tf.global_variables_initializer())
        # define saver
        saver = tf.train.Saver()

        lr = lr_start
        for epoch in range(max_epochs):
            epoch_time = time.time()

            # Split data into batches
            batch_count = trn_x.shape[0] / batch_size
            rnd_idx = np.random.permutation(trn_x.shape[0])
            batch_x = np.split(trn_x[rnd_idx[:batch_count * batch_size]], batch_count)
            batch_y = np.split(trn_y[rnd_idx[:batch_count * batch_size]], batch_count)

            for m in range(batch_count):
                # apply one-hot encoding
                one_hot = np.eye(num_feats)[batch_y[m]]
                # train the network on that batch
                loss1, _, acc1 = sess.run([loss, train_op, accuracy], feed_dict={input: batch_x[m],
                                                                            labels: one_hot,
                                                                            learning_rate: lr})
                loss_batch.append(loss1)
                acc_batch.append(acc1)

            # decrease learning rate by a factor every certain amount of epochs
            if epoch in lr_strat:
                lr *= lr_factor
            # print information on how training is going
            if epoch % print_epoch == 0:
                print("Epoch {}: loss {} -- train acc {} -- time {}".format(epoch,
                                                                            np.mean(loss_batch),
                                                                            np.mean(acc_batch),
                                                                            time.time() - epoch_time))
                loss_batch = []
                acc_batch = []

            # save model
            saver.save(sess, os.path.join(path_experiment, '_model.ckpt'), global_step=epoch)


def test_net(data_x, data_y):
    test_time = time.time()
    tf.reset_default_graph()
    # define placeholder and network
    input = tf.placeholder(tf.float32, [batch_size, data_x.shape[1], data_x.shape[2], data_x.shape[3]])
    if network == 'simple':
        net = Simple(input)
    elif network == 'lenet':
        net = LeNet(input, num_features=num_feats)
    # start session
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        # Load trained model
        checkpoint_dir = path_experiment
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Error: no checkpoint found for trained model')
        # extract predictions
        pred = np.zeros(data_x.shape[0]).astype(int)
        for m in range(data_x.shape[0]/batch_size):
            all_probs = net.probs.eval({input: data_x[m*batch_size:(m+1)*batch_size]})
            pred[m*batch_size:(m+1)*batch_size] = np.argmax(all_probs,1)
    # evaluate accuracy
    acc_test = (100.0 * np.sum(pred == data_y)) / float(pred.shape[0])
    print('---')
    print ("Accuracy {} -- time {}".format(acc_test, time.time() - test_time))


if __name__ == '__main__':
    # Check if experiment folder already exists
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)

    # Load dataset
    mnist = umnist.load_mnist_32x32(verbose=False)

    # Train
    train_net(mnist.train.images, mnist.train.labels)

    # Test
    test_net(mnist.test.images, mnist.test.labels)
