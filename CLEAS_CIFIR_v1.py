from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import data_load_cifar as dl
import pickle
from CLEAS_CNN import evaluate
# in this version, I will use a multi-layer network and desgin a RNN controller.
# read dataset
X_train_tasks=dl.Task_train_x
Y_train_tasks=dl.Task_train_y
X_test_tasks=dl.Task_test_x
Y_test_tasks=dl.Task_test_y
X_vali_tasks=dl.Task_vali_x
Y_vali_tasks=dl.Task_vali_y

def ops(task_vars,old_action,data,task_id):
    g2=tf.Graph()
    sess2=tf.Session(graph=g2)
    with sess2.as_default():
        with g2.as_default():
            batch_size = 1
            hidden_size = 64
            NUM_NU = 128
            output_size = 2
            states = tf.placeholder(tf.int32, [3, 128], name="states")
            em_states = tf.placeholder(tf.float32, [128 * 3, output_size], name="em_states")
            index_hot = tf.constant(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], tf.float32)
            def weight_variablec(shape):
                initial = tf.truncated_normal(shape, stddev=0.01)
                return tf.Variable(initial)
            def bias_variablec(shape):
                initial = tf.constant(0.0, shape=shape)
                return tf.Variable(initial)
            def policy_network_2(states, reuse=False):
                with tf.variable_scope("policy_network2", reuse=reuse):
                    nas_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
                    input_s = states
                    input_em = input_s
                    outputs, state = tf.nn.dynamic_rnn(
                        nas_cell,
                        input_em,
                        dtype=tf.float32
                    )
                    PW = weight_variablec([hidden_size, output_size])
                    Pb = bias_variablec([output_size])
                    # print 'input', input_em
                    # print("outputs: ", outputs)
                    tensor = outputs
                    outputs = (tf.matmul(outputs, [PW] * batch_size) + Pb)  # tf.nn.tanh  + Pb
                    return outputs, state, tensor
            index_1 = tf.constant([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], tf.float32)
            index = tf.tile(index_1, [NUM_NU, 1, 1])
            index = tf.transpose(index, [1, 0, 2])
            embeddings_states = tf.nn.embedding_lookup(index_hot, states)  # index_hot
            print embeddings_states, index, states
            HS = tf.concat([embeddings_states, index], axis=2)
            HS = tf.reshape(HS, [3 * NUM_NU, 8])
            HS = tf.expand_dims(HS, 0)
            action, en_state, last = policy_network_2(HS)
            policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network2")
            CNN_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="CNN")
            action = tf.nn.softmax(tf.identity(action, name="action_scores"))
            logprobs, _, _ = (policy_network_2(HS, reuse=True))
            #print '--->>>>>logpro', logprobs[:, -1, :], logprobs
            discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logprobs[0], labels=em_states)
            #print 'loss', tf.shape(cross_entropy_loss)
            cross_entropy_loss = tf.reduce_sum(cross_entropy_loss)
            # print policy_network_variables
            reg_loss = tf.reduce_sum(
                [tf.reduce_sum(tf.square(xs)) for xs in policy_network_variables])  # Regularization
            loss = cross_entropy_loss + reg_loss * 0.001
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(0.01, global_step,
                                                       500, 0.96, staircase=True)
            # learning_rate
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)  # AdamOptimizer RMSPropOptimizer
            # compute gradients
            gradients_at = optimizer.compute_gradients(loss, var_list=policy_network_variables)
            # compute policy gradients
            for i, (grad, var) in enumerate(gradients_at):
                if grad is not None:
                    gradients_at[i] = (grad * discounted_rewards, var)
            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                train_op = optimizer.apply_gradients(gradients_at, global_step=global_step)
            print("training the new task", task_id)
            var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            init = tf.variables_initializer(var_lists)
            sess2.run(init)
            np.random.seed(2)
            neu_1 = []
            neu_2 = []
            neu_3 = []
            neuset=[]
            episode = 200
            exploration = np.random.binomial(1, 0.3)
            result_acc = []
            total_reward = 0.0
            ACC_regard = []
            pre_acc = 0.0
            best_acc = 0
            reward_buffer = []
            state_buffer = []
            def storeRollout(state, reward):
                reward_buffer.append(reward)
                state_buffer.append(state)
            for iter in range(episode):  # hypernetwork
                if exploration == 1 or iter == 0:
                    state_in = (np.random.randint(low=0, high=4, size=[3, NUM_NU]).astype(np.int32))
                    print 'random, state----------------------------------------'
                options = []
                # print 'state',state_in
                action_ = action.eval(feed_dict={states: state_in})  # get action
                # print action_
                actions = []
                real_action = []
                action_ = action_[0][:, 0]  # obtain first c
                action_ = [action_[xp:xp + NUM_NU] for xp in range(0, len(action_), NUM_NU)]
                # if exploration == 1:
                #     action_ = (np.random.randint(low=0, high=4, size=[3, NUM_NU]).astype(np.int32))
                for i in range(len(action_)):
                    option = []
                    opt = []
                    for j in range(len(action_[i])):
                        value = np.random.choice([0, 1], p=[action_[i][j],1- action_[i][j]])
                        # print value,action_[i][j]
                        #value=action_[i][j]
                        if value > 0.5:
                            option.append(j)
                            actions.append(1)
                            opt.append(j)
                        else:
                            actions.append(0)
                            option.append(NUM_NU)
                    options.append(option)
                    real_action.append(len(opt))
                # print 'action',actions
                print 'episode', iter
                print 'option', real_action
                neu_1.append((real_action[0]))
                neu_2.append((real_action[1]))
                neu_3.append((real_action[2]))
                neuset.append(real_action)
                old_action = old_action  # previous task
                # print('old_action',old_action)
                old_opt = old_action
                # print old_opt
                old_actions = []
                rate = 0
                up_num = 10
                new_options = []
                num_index = []
                for xs in range(len(old_opt)):  # num of lyaer
                    temp = []  # update parametrs
                    temp_opt = []  # options
                    index = 0
                    real_index = 0
                    for ys in range(len(old_opt[xs])):
                        if old_opt[xs][ys] == NUM_NU:
                            index += 1
                            if index <= up_num:
                                temp.append(options[xs][ys])
                                temp_opt.append(options[xs][ys])
                                if options[xs][ys] is not NUM_NU:
                                    real_index += 1
                            else:
                                temp.append(NUM_NU)
                                temp_opt.append(NUM_NU)
                        else:
                            if options[xs][ys] == NUM_NU:
                                temp.append(options[xs][ys])
                                temp_opt.append(options[xs][ys])
                            else:
                                temp.append(NUM_NU)
                                temp_opt.append(options[xs][ys])

                    old_actions.append(temp)  # update paras
                    new_options.append(temp_opt)
                    num_index.append(real_index)
                old_opts = np.array(old_actions)
                print 'shape', np.shape(old_opts), num_index
                options = new_options
                global_opt = []
                for xs in range(len(old_opt)):
                    temp = []
                    for ys in range(len(old_opt[xs])):
                        if old_opt[xs][ys] == options[xs][ys]:
                            temp.append(options[xs][ys])
                            rate += 1
                        else:
                            if old_opt[xs][ys] == NUM_NU:
                                temp.append(options[xs][ys])
                            else:
                                temp.append(old_opt[xs][ys])
                    global_opt.append(temp)
                evaluate_task = evaluate(NUM_NU)
                vali_acc, test_acc, new_taskvars = evaluate_task.model(task_vars, data, options, old_opts)
                act = []
                mask_in = []
                for ls in options:
                    temp = []
                    for v in ls:
                        if v == NUM_NU:
                            mask_in.append(0.0)
                        else:
                            temp.append(v)
                            mask_in.append(1.0)
                    act.append(temp) #decide which neuron will be selected in the current task
                rewards = vali_acc - 0.001 * np.sum(num_index)
                result_acc.append(test_acc)  #
                if rewards > best_acc:
                    best_acc = rewards
                    best_choose=[[len(act[0]),len(act[1])],[test_acc],[num_index]]
                    best_actions=global_opt
                    best_vars=new_taskvars
                    best_accs=test_acc
                    np.savetxt('./choose/cifar_action'+str(task_id)+'.txt', np.array(global_opt))  # the allocated neurons for all task
                    np.savetxt('./choose/cifar_option'+str(task_id)+'.txt', np.array(options))  # the neurons for current task
                Areward_ = rewards  # *acc3#(acc_a)#*reward_)
                print('rewards-->', rewards, best_acc,best_accs)
                if iter==0:
                    reward=Areward_
                else:
                    reward = Areward_- ema(reward_buffer)#*0.49# * 0.9 #0.49
                for ai in real_action:
                    if ai == 0:
                        reward = -1
                        sess2.run(init)
                total_reward += reward
                ACC_regard.append(Areward_)
                print 'reward', reward, 'episode', iter
                storeRollout(actions, reward)
                new_states = np.array(state_buffer[-1:])
                #print np.shape(new_states)
                rewars = reward_buffer[-1:]
                # print new_states
                input_emstate = np.array(
                    [new_states[0][xp:xp + NUM_NU] for xp in range(0, len((new_states[0])), NUM_NU)])
                state_in = input_emstate  # next state
                #print np.shape(input_emstate)
                y_state = []
                for i in range(len(new_states[0])):
                    if new_states[0][i] == 0:
                        y_state.append([0.0, 1.0])
                    else:
                        y_state.append([1.0, 0.0])
                _, loss_, re_loss = sess2.run([train_op, loss, reg_loss],
                                             feed_dict={states: input_emstate, em_states: y_state,
                                                        discounted_rewards: rewars})
            print(result_acc)
            print('best choose',best_choose)
            sess2.close()
            #draw_pic_metric_ACC(result_acc)

        return best_vars,best_actions,best_accs
def ema(values):
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]
    return a[-1]
def basic_ops(train_sess,data):
    with tf.variable_scope('first_task_network'):
        #tf.set_random_seed(0)
        #np.random.seed(2)
        x = tf.placeholder(tf.float32, shape=[None, 3072])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        NUM_NU = 128
        OLD_ACTION_SET = tf.placeholder(tf.int32, [3, NUM_NU])
        ACTION_SET = tf.placeholder(tf.int32, [3, NUM_NU])
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.06)  # tf.random_normal(shape)#
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.0, shape=shape)
            return tf.Variable(initial)
        # define network of classification-----------
        with tf.variable_scope('CNN'):
            conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, NUM_NU], mean=0, stddev=0.08))
            conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, NUM_NU, NUM_NU], mean=0, stddev=0.08))
            conv3_filter = tf.Variable(tf.truncated_normal(shape=[4, 4, NUM_NU, NUM_NU], mean=0, stddev=0.08))
        with tf.variable_scope('MLP'):
            W_1 = weight_variable([2048, 120])  # layer one
            b_1 = bias_variable([120])
            W_2 = weight_variable([120, 84])  # layer two
            b_2 = bias_variable([84])
            W_o = weight_variable([84, 10])  # layer three
            b_o = bias_variable([10])
        print 'c1', conv1_filter
        print 'shapes', tf.transpose(conv1_filter, [3, 0, 1, 2])[0]
        x_s = tf.reshape(x, [tf.shape(x)[0], 32, 32, 3])#convert data

        conv1 = tf.nn.conv2d(x_s, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv1)
        C1 = tf.transpose(conv1, [3, 0, 1, 2])
        conv1 = tf.transpose(tf.gather(C1, ACTION_SET[0]), [1, 2, 3, 0])
        print 'c1', conv1
        conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv1_bn = tf.layers.batch_normalization(conv1_pool)

        conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(conv2)
        print 'c2', conv2
        C2 = tf.transpose(conv2, [3, 0, 1, 2])
        conv2 = tf.transpose(tf.gather(C2, ACTION_SET[1]), [1, 2, 3, 0])
        conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv2_bn = tf.layers.batch_normalization(conv2_pool)

        conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.relu(conv3)
        print 'cn3', conv3
        C3 = tf.transpose(conv3, [3, 0, 1, 2])
        conv3 = tf.transpose(tf.gather(C3, ACTION_SET[2]), [1, 2, 3, 0])

        conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv3_bn = tf.layers.batch_normalization(conv3_pool)
        print 'c3', conv3_bn
        flat = tf.layers.flatten(conv3_bn)
        print 'flat', flat

        fc1 = tf.nn.relu(tf.matmul(flat, W_1) + b_1)
        fc2 = tf.nn.relu(tf.matmul(fc1, W_2) + b_2)
        y = tf.matmul(fc2, W_o) + b_o
        cnn_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="first_task_network/CNN")
        var_list = cnn_variables  # collect parameters
        #print(var_list)
        mlp_list = [W_1, W_2, W_o, b_1, b_2, b_o]
        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(1e-3, global_step,
        #                                            500, 0.96, staircase=True)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        optimizer_task = tf.train.AdamOptimizer(1e-3)
        net_gradients = optimizer_task.compute_gradients(cross_entropy, var_list + mlp_list)
        mlp_gradients = net_gradients[3:]  # tf.gradients(cross_entropy,mlp_list)
        # actions---
        #print('shape',net_gradients)
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

        global_vars_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # print('task network',vars_lists)
        init = tf.variables_initializer(global_vars_lists)
        train_sess.run(init)

        action1 = [1.0] * 16 + [0.0] * 112
        action2 = [1.0] * 16 + [0.0] * 112
        action3 = [1.0] * 16 + [0.0] * 112
        actions = []
        real_action = []
        options = []
        action_ = [action1, action2, action3]
        for i in range(len(action_)):
            option = []
            opt = []
            for j in range(len(action_[i])):
                # value = np.random.choice([0, 1], p=[action_[i][j],1- action_[i][j]])
                # print value,action_[i][j]
                if action_[i][j] > 0.5:
                    option.append(j)
                    actions.append(1)
                    opt.append(j)
                else:
                    actions.append(0)
                    option.append(NUM_NU)

            options.append(option)
            real_action.append(len(opt))

        neu_1 = []
        neu_2 = []
        neu_3 = []
        neuset = []
        print 'option', real_action
        # OP=[len(options[0]),len(options[1]),len(options[2]),len(options[3])]
        neu_1.append((real_action[0]))
        neu_2.append((real_action[1]))
        neu_3.append((real_action[2]))
        neuset.append(real_action)
        EPOCH=2000
        train_x = np.array(data[0][0])
        train_y=np.array(data[0][1])
        for iter in range(EPOCH):
            batch_x, batch_y = dl.next_batch(train_x, train_y, 100)
            _, loss_c = train_sess.run([train_step_, cross_entropy],
                                       feed_dict={x: batch_x, y_: batch_y, ACTION_SET: options,
                                                  OLD_ACTION_SET: options})
        print('---',len(data[1][0]))
        val= accuracy.eval(
            feed_dict={x: data[1][0], y_: data[1][1], ACTION_SET: options})  # task-A
        acc_test = accuracy.eval(
            feed_dict={x: data[2][0], y_: data[2][1], ACTION_SET: options})  # task-A
        print('validation accuracy', val)
        print('test accuracy', acc_test)
        var = train_sess.run([conv1_filter, conv2_filter, conv3_filter])
    return var,options,acc_test
def draw_pic_metric_ACC(Acc,name='ACC'):
    width = 8
    height =8
    plt.figure(figsize=(width, height))
    train_axis = np.array(range(1, len(Acc) + 1, 1))
    #print(train_axis)
    #print(Acc)
    plt.plot(train_axis, Acc, 'r*-')
    #plt.ylim(0, 100)  # y
    plt.title(name)
    #plt.legend(loc='best', shadow=True)
    plt.ylabel('numbers')
    plt.xlabel(' iterations')
    plt.show()
def train():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    dataset=[]
    for i in range(10):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset1 = [train_x, train_y]
    testset1 = [test_x, test_y]
    valiset1 = [vali_x, vali_y]
    #print(np.shape(vali_x))
    dataset.append([trainset1,valiset1,testset1])
    print('just a test',len(dataset[0][1][0]))
    # Task2
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    for i in range(10, 20):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset2 = [train_x, train_y]
    testset2 = [test_x, test_y]
    valiset2 = [vali_x, vali_y]
    dataset.append([trainset2,valiset2,testset2])
    # Task3
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    for i in range(20, 30):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset3 = [train_x, train_y]
    testset3 = [test_x, test_y]
    valiset3 = [vali_x, vali_y]
    dataset.append([trainset3,valiset3,testset3])
    # Task4
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    for i in range(30, 40):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset4 = [train_x, train_y]
    testset4 = [test_x, test_y]
    valiset4 = [vali_x, vali_y]
    dataset.append([trainset4,valiset4,testset4])
    # Task5
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    for i in range(40, 50):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset5 = [train_x, train_y]
    testset5 = [test_x, test_y]
    valiset5= [vali_x, vali_y]
    dataset.append([trainset5,valiset5,testset5])
    # Task6
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    for i in range(50, 60):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset6 = [train_x, train_y]
    testset6 = [test_x, test_y]
    valiset6= [vali_x, vali_y]
    dataset.append([trainset6,valiset6,testset6])
    # Task7
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    for i in range(60, 70):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset7 = [train_x, train_y]
    testset7 = [test_x, test_y]
    valiset7 = [vali_x, vali_y]
    dataset.append([trainset7,valiset7,testset7])
    # Task8
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    for i in range(70, 80):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset8 = [train_x, train_y]
    testset8 = [test_x, test_y]
    valiset8= [vali_x, vali_y]
    dataset.append([trainset8,valiset8,testset8])
    # Task9
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    for i in range(80, 90):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset9= [train_x, train_y]
    testset9= [test_x, test_y]
    valiset9 = [vali_x, vali_y]
    dataset.append([trainset9,valiset9,testset9])
    # Task10
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    vali_x = []
    vali_y = []
    for i in range(90, 100):
        train_x = train_x + X_train_tasks[i]
        train_y = train_y + Y_train_tasks[i]
        test_x = test_x + X_test_tasks[i]
        test_y = test_y + Y_test_tasks[i]
        vali_x=vali_x+X_vali_tasks[i]
        vali_y = vali_y + Y_vali_tasks[i]
    trainset10 = [train_x, train_y]
    testset10 = [test_x, test_y]
    valiset10=[vali_x, vali_y]
    dataset.append([trainset10,valiset10,testset10])
    print('the number of tasks:',len(dataset))
    BEST_ACCS=[]
    for i in range(len(dataset)):
        task_id=i
        data=dataset[i]
        if task_id==0:
            g1=tf.Graph()
            sess1=tf.Session(graph=g1)
            with sess1.as_default():
                with g1.as_default():
                    task_vars,old_action,acc = basic_ops(sess1, data)
                    sess1.close()
                    BEST_ACCS.append(acc)
                    f = open('./vars/cifar-task_' + str(task_id+1), 'wb') #store the first task
                    pickle.dump(task_vars, f)
                    f.close()
                    np.savetxt('./results/cifar_task1_action' + str(task_id + 1) + '.txt', np.array(old_action))
        else:
            tf.reset_default_graph()
            task_vars,old_action,best_acc=ops(task_vars, old_action,data, task_id)
            BEST_ACCS.append(best_acc)
            print('task best accuracy',task_id,best_acc)
            # if task_id>=1:
            #     break
    print('vag_acc',np.mean(BEST_ACCS))
train()