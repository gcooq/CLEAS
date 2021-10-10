import tensorflow as tf
from data_load_cifar import next_batch

class evaluate:
    def __init__(self, NUM_NU):
        self.NUM_NU=NUM_NU
    def model(self,vars,data,new_options,old_options):
        #tf.reset_default_graph()
        g = tf.Graph()
        sess = tf.Session(graph=g)
        with sess.as_default():
            with g.as_default():
                self.sess = sess
                with tf.variable_scope('task_network'):
                    x = tf.placeholder(tf.float32, shape=[None, 3072])
                    y_ = tf.placeholder(tf.float32, shape=[None, 10])
                    OLD_ACTION_SET = tf.placeholder(tf.int32, [3, self.NUM_NU])
                    ACTION_SET = tf.placeholder(tf.int32, [3, self.NUM_NU])

                    def weight_variable(shape):
                        initial = tf.truncated_normal(shape, stddev=0.06)  # tf.random_normal(shape)#
                        return tf.Variable(initial)

                    def bias_variable(shape):
                        initial = tf.constant(0.0, shape=shape)
                        return tf.Variable(initial)

                    # define network of classification-----------
                    with tf.variable_scope('CNN'):
                        conv1_filter = tf.Variable(vars[0])
                        conv2_filter = tf.Variable(vars[1])
                        conv3_filter = tf.Variable(vars[2])
                    with tf.variable_scope('MLP'):
                        W_1 = weight_variable([2048, 120])  # layer one
                        b_1 = bias_variable([120])
                        W_2 = weight_variable([120, 84])  # layer two
                        b_2 = bias_variable([84])
                        W_o = weight_variable([84, 10])  # layer three
                        b_o = bias_variable([10])
                    x_s = tf.reshape(x, [tf.shape(x)[0], 32, 32, 3])  # convert data
                    conv1 = tf.nn.conv2d(x_s, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
                    conv1 = tf.nn.relu(conv1)
                    C1 = tf.transpose(conv1, [3, 0, 1, 2])
                    conv1 = tf.transpose(tf.gather(C1, ACTION_SET[0]), [1, 2, 3, 0])
                    # print 'c1', conv1
                    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
                    conv1_bn = tf.layers.batch_normalization(conv1_pool)
                    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
                    conv2 = tf.nn.relu(conv2)
                    # print 'c2', conv2
                    C2 = tf.transpose(conv2, [3, 0, 1, 2])
                    conv2 = tf.transpose(tf.gather(C2, ACTION_SET[1]), [1, 2, 3, 0])
                    conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
                    conv2_bn = tf.layers.batch_normalization(conv2_pool)
                    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
                    conv3 = tf.nn.relu(conv3)
                    # print 'cn3', conv3
                    C3 = tf.transpose(conv3, [3, 0, 1, 2])
                    conv3 = tf.transpose(tf.gather(C3, ACTION_SET[2]), [1, 2, 3, 0])
                    conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
                    conv3_bn = tf.layers.batch_normalization(conv3_pool)
                    # print 'c3', conv3_bn
                    flat = tf.layers.flatten(conv3_bn)
                    # print 'flat', flat
                    fc1 = tf.nn.relu(tf.matmul(flat, W_1) + b_1)
                    fc2 = tf.nn.relu(tf.matmul(fc1, W_2) + b_2)
                    y = tf.matmul(fc2, W_o) + b_o
                    cnn_variables = [conv1_filter, conv2_filter,
                                     conv3_filter]  # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="task_network/CNN")
                    var_list = cnn_variables  # collect parameters
                    #print(var_list)
                    mlp_list = [W_1, W_2, W_o, b_1, b_2, b_o]
                    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
                    optimizer_task = tf.train.AdamOptimizer(1e-3)
                    net_gradients = optimizer_task.compute_gradients(cross_entropy, var_list + mlp_list)
                    mlp_gradients = net_gradients[3:]  # tf.gradients(cross_entropy,mlp_list)
                    # actions---
                    #print('shape', net_gradients[0])
                    S1 = tf.transpose(net_gradients[0][0], [3, 0, 1, 2])
                    cnn_1 = (tf.transpose(tf.gather(S1, OLD_ACTION_SET[0]), [1, 2, 3, 0]), net_gradients[0][1])

                    S2 = tf.transpose(net_gradients[1][0], [3, 0, 1, 2])
                    cnn_2 = tf.transpose(tf.gather(S2, OLD_ACTION_SET[1]), [1, 2, 3, 0])
                    S22 = tf.transpose(cnn_2, [2, 0, 1, 3])
                    cnn_2 = (tf.transpose(tf.gather(S22, OLD_ACTION_SET[0]), [1, 2, 0, 3]), net_gradients[1][1])

                    S3 = tf.transpose(net_gradients[2][0], [3, 0, 1, 2])
                    cnn_3 = tf.transpose(tf.gather(S3, OLD_ACTION_SET[2]), [1, 2, 3, 0])
                    S32 = tf.transpose(cnn_3, [2, 0, 1, 3])
                    cnn_3 = (tf.transpose(tf.gather(S32, OLD_ACTION_SET[1]), [1, 2, 0, 3]), net_gradients[2][1])

                    var_list_new = optimizer_task.apply_gradients([cnn_1, cnn_2, cnn_3] + mlp_gradients)
                    train_step_ = var_list_new
                    # performance metrics
                    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    self.sess.run(tf.global_variables_initializer())  # initialize adam
                    old_opt = old_options
                    new_opt = new_options
                    EPOCH = 200#2000
                    train_x = data[0][0]
                    train_y = data[0][1]
                    for iter in range(EPOCH):
                        batch_x, batch_y = next_batch(train_x, train_y, 100)
                        _, loss_c = self.sess.run([train_step_, cross_entropy],
                                                  feed_dict={x: batch_x, y_: batch_y, ACTION_SET: new_opt,
                                                             OLD_ACTION_SET: old_opt})
                    print('---', len(data[1][0]))
                    val = accuracy.eval(
                        feed_dict={x: data[1][0], y_: data[1][1], ACTION_SET: new_opt})
                    acc_test = accuracy.eval(
                        feed_dict={x: data[2][0], y_: data[2][1], ACTION_SET: new_opt})
                    print('validation accuracy', val)
                    print('test accuracy', acc_test)
                    self.vars = self.sess.run([conv1_filter, conv2_filter, conv3_filter])
                self.sess.close()
        #with tf.Session() as sess:

        return val,acc_test,self.vars