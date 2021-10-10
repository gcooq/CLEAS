import tensorflow as tf


class evaluate:
    def __init__(self, NUM_NU, max_layers=2):
        self.NUM_NU=NUM_NU
        self.max_layers=max_layers

    def model(self,vars,data,new_options,old_options):
        # define hyper network which is fixed.
        # with tf.variable_scope('task_network'):
        with tf.Session() as sess:
            self.sess=sess
            with tf.variable_scope('task_network'):
                # define input andtarget placeholders
                x = tf.placeholder(tf.float32, shape=[None, 784])
                y_ = tf.placeholder(tf.float32, shape=[None, 10])
                ACTION_SET = tf.placeholder(tf.int32, [2, self.NUM_NU])
                OLD_ACTION_SET = tf.placeholder(tf.int32, [2, self.NUM_NU])

                # variable initialization functions
                def weight_variable(shape):
                    initial = tf.truncated_normal(shape, stddev=0.08)
                    return tf.Variable(initial)

                def bias_variable(shape):
                    initial = tf.constant(0.1, shape=shape)
                    return tf.Variable(initial)


                W1 = tf.Variable(vars[0])
                b1 =  tf.Variable(vars[1])
                W2 = tf.Variable(vars[2])
                b2 = tf.Variable(vars[3])
                W3 =weight_variable([self.NUM_NU, 10])#tf.Variable(vars[4])  #
                b3 =bias_variable([10])#tf.Variable(vars[5]) #

                h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer 1
                h1s = tf.transpose(h1, [1, 0])
                h1s = tf.transpose((tf.gather(h1s, ACTION_SET[0])))  # operation on hidden layer b

                h2 = tf.nn.relu(tf.matmul(h1s, W2) + b2)  # hidden layer 2
                h2s = tf.transpose(h2, [1, 0])
                h2s = tf.transpose((tf.gather(h2s, ACTION_SET[1])))

                y = tf.matmul(h2s, W3) + b3  # multihead

                var_list = [W1, W2, W3, b1, b2, b3]  # collect parameters

                # vanilla single-task loss
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) \
                                + 0.0001 * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
                optimizer_task = tf.train.AdamOptimizer(1e-3)
                up_grads = optimizer_task.compute_gradients(cross_entropy, var_list)  # compute the gradients

                W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad = up_grads
                # print 'set',up_grads
                W_u = [W1_grad, W2_grad]  # project to the hidden layer

                new_W = []

                for v in range(self.max_layers):
                    if v == 0:  # first layer
                        S = tf.transpose(W_u[v][0])
                        new_W.append((tf.transpose(tf.gather(S, OLD_ACTION_SET[v])), W_u[v][1]))
                        #print(OLD_ACTION_SET[v])
                        up_b1 = (tf.gather(b1_grad[0], OLD_ACTION_SET[v]), b1_grad[1])
                    else:  # second layer
                        s1 = tf.transpose(W_u[v][0])
                        s2 = W_u[v][0]
                        v1 = tf.gather(s1, OLD_ACTION_SET[v - 1])
                        v1 = tf.transpose(v1)
                        v2 = tf.gather(s2, OLD_ACTION_SET[v])
                        new_W.append((tf.sqrt(v1 * v2), W_u[v][1]))
                        up_b2 = (tf.gather(b2_grad[0], OLD_ACTION_SET[v]), b2_grad[1])
                w1_grad = new_W[0]
                w2_grad = new_W[1]
                # the output layer will always be optimized
                w3_grad = W3_grad
                up_b3 = b3_grad
                # print [w1_grad,w2_grad,w3_grad,up_b1,up_b2,up_b3]

                var_list_new = optimizer_task.apply_gradients([w1_grad, w2_grad, w3_grad, up_b1, up_b2, up_b3])
                train_step_ = var_list_new

                # performance metrics
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                old_opt = old_options
                new_opt = new_options
                self.sess.run(tf.global_variables_initializer()) #initialize adam
                for step in range(2000):
                    train_batch = data.train.next_batch(100)
                    _, batch_loss = self.sess.run([train_step_, cross_entropy],
                                                   feed_dict={x: train_batch[0], y_: train_batch[1],
                                                              ACTION_SET: new_opt,
                                                              OLD_ACTION_SET: old_opt}
                                                   )
                acc_test = accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels,
                                                    ACTION_SET: new_opt})  # task-A
                val = accuracy.eval(feed_dict={x: data.validation.images, y_: data.validation.labels,
                                               ACTION_SET: new_opt})  # task-A
                print('validation accuracy', val)
                print('test accuracy', acc_test)
                self.vars=self.sess.run([W1, b1, W2, b2, W3, b3])
        return val,acc_test,self.vars




