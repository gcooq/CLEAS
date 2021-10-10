from __future__ import division
import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import datetime
import random
from mnist_permulation import *
import  time
#from CLEAS_MLP import *
# from CLEAS_controller import *

#hyper-parameters
episode=200
# savers = tf.train.Saver()
from CLEAS_MLP import evaluate
def ema(values):
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]
    return a[-1]
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
def ops(task_vars,old_action,data,task_id):
    g2=tf.Graph()
    sess2=tf.Session(graph=g2)
    with sess2.as_default():
        with g2.as_default():
            batch_size = 1
            hidden_size = 32  #32
            NUM_NU = 1000
            # variable initialization functions
            def weight_variablec(shape):
                initial = tf.truncated_normal(shape, stddev=0.01)
                return tf.Variable(initial)

            def bias_variablec(shape):
                initial = tf.constant(0.01, shape=shape)
                return tf.Variable(initial)
            index_hot = tf.constant([[0.0, 1.0], [1.0, 0.0]], tf.float32)
            # actor
            #tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            #tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
            states = tf.placeholder(tf.int32, [2, NUM_NU], name="states")
            #mask = tf.placeholder(tf.float32, [1, 2 * NUM_NU], name="mask")
            em_states = tf.placeholder(tf.float32, [2 * NUM_NU, 2], name="em_states")
            def policy_network_2(states, reuse=False):
                with tf.variable_scope("policy_network2", reuse=reuse):
                    # nas_cell = tf.contrib.rnn.NASCell(64)
                    nas_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
                    # cell_state=nas_cell.zero_state(batch_size=1,dtype=tf.float32)
                    input_s = states
                    #print input_s
                    input_em = input_s  # tf.nn.embedding_lookup(initial_embeddings,input_s)
                    outputs, state = tf.nn.dynamic_rnn(
                        nas_cell,
                        input_em,
                        dtype=tf.float32
                    )
                    # bias = tf.Variable([0.05] * 50 * max_layers)
                    PW = weight_variablec([hidden_size, 2])
                    Pb = bias_variablec([2])
                    #print 'input', input_em
                    #print("outputs: ", outputs)
                    tensor = outputs
                    outputs = (tf.matmul(outputs, [PW] * batch_size) + Pb)  # tf.nn.tanh  + Pb
                    return outputs, state, tensor
            index_1 = tf.constant([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], tf.float32) #layer identification
            index = tf.tile(index_1, [NUM_NU, 1, 1])
            index = tf.transpose(index, [1, 0, 2])
            embeddings_states = tf.nn.embedding_lookup(index_hot, states)
            HS = tf.concat([embeddings_states, index], axis=2)
            HS = tf.reshape(HS, [2 * NUM_NU, 6])
            HS = tf.expand_dims(HS, 0)
            print 'HS', HS
            action, en_state, last = policy_network_2(HS)
            policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network2")
            #print 'ac', action
            action = tf.nn.softmax(tf.identity(action, name="action_scores"))
            # action = tf.cast(tf.scalar_mul(100.0, action), tf.int32, name="predicted_action")
            logprobs, _, _ = (policy_network_2(HS, reuse=True))
           # print '--->>>>>logpro', logprobs[:, -1, :], logprobs
            discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
            cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logprobs[0], labels=em_states)
            #print 'loss', tf.shape(cross_entropy_loss)
            cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)  # mask
            # print policy_network_variables
            reg_loss = tf.reduce_sum(
                [tf.reduce_sum(tf.square(xs)) for xs in policy_network_variables])  # Regularization
            loss = cross_entropy_loss  # *100+reg_loss*0.001
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
            #print 'global var_lists', var_lists
            init = tf.variables_initializer(var_lists)
            seed=4
            setup_seed(seed)
            #sess2.run(tf.set_random_seed(seed))
            sess2.run(init)
            #np.random.seed(2)
            print('randoms---',np.random.random())
            neuset = []
            neu_1 = []
            neu_2 = []
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
                    state_in = (np.random.randint(low=0, high=2, size=[2, NUM_NU]).astype(np.float32))
                    print(' random state')
                options = []
                action_ = action.eval(feed_dict={states: state_in})  # get action
                actions = []
                real_action = []
                action_ = action_[0][:, 0]  # obtain first c
                action_ = [action_[xp:xp + NUM_NU] for xp in range(0, len(action_), NUM_NU)]  # design the network
                # print(action_)
                for i in range(len(action_)):
                    option = []
                    opt = []
                    for j in range(len(action_[i])):
                        # print action_[i][j]
                        #value = np.random.choice([1, 0], p=[action_[i][j], 1 - action_[i][j]])

                        value=action_[i][j]
                        #print action_[i][j]
                        if value > 0.5:
                            option.append(j)
                            actions.append(1)
                            opt.append(j)
                        else:
                            actions.append(0)
                            option.append(NUM_NU)
                    options.append(option)
                    real_action.append(len(opt))
                #print(actions)  # bool output
                #print(options)  # index
                #print('the current network architecture', real_action)
                neu_1.append((real_action[0]))
                neu_2.append((real_action[1]))
                neuset.append(real_action)
                old_action=old_action #previous task
                #print('old_action',old_action)
                old_opt = old_action
                # print old_opt
                old_actions = []
                rate = 0
                up_num=30 #control the scale of added new neurons
                num_index=[]
                new_options=[]
                for xs in range(len(old_opt)): #num of lyaer
                    temp=[]#update parametrs
                    temp_opt=[]#options
                    index=0
                    real_index=0
                    for ys in range(len(old_opt[xs])):
                        if old_opt[xs][ys]==NUM_NU: #arrage new
                            index+=1
                            if index<=up_num:
                                temp.append(options[xs][ys])
                                temp_opt.append(options[xs][ys])
                                if options[xs][ys] is not NUM_NU:
                                    real_index+=1
                            else:
                                temp.append(NUM_NU)
                                temp_opt.append(NUM_NU)
                        else:
                            if options[xs][ys]==NUM_NU: #
                                temp.append(options[xs][ys])
                                temp_opt.append(options[xs][ys])
                            else:
                                temp.append(NUM_NU)
                                temp_opt.append(options[xs][ys])
                    old_actions.append(temp)  # update paras
                    new_options.append(temp_opt) # all chooses
                    num_index.append(real_index)
                old_opts = np.array(old_actions) # the neurons can be optimized
                #print 'shape', np.shape(old_opts) # the nerons can
                options=new_options #!!!!! #this means which weights can be used in current task
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
                    global_opt.append(temp) #this means the neurons can be selected finally.
                # print('old neurons', list(old_opts))#can be updated
                # np.savetxt('./temp/otimized_options.dat', np.array(old_opts).astype(int))
                # print('the used neurons for current task',list(options))
                # np.savetxt('./temp/options.dat',list(options))#the used neurons for current task
                # print('allocated neurons for all tasks',list(global_opt)) #actions id number
                # np.savetxt('./temp/current_options.dat', np.array(global_opt).astype(int)) #allocated neurons for all tasks
                evaluate_task=evaluate(NUM_NU,max_layers=2)
                vali_acc,test_acc,new_taskvars=evaluate_task.model(task_vars,data,options,old_opts)
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
                rewards = vali_acc - 0.0001 * np.sum(num_index)  # * (1 - 0.1 * np.mean(real_action) / NUM_NU)
                result_acc.append(test_acc)  # ,acc2,acc3,acc4,acc5,acc6,acc7,acc8,acc9,acc10
                if rewards > best_acc:
                    best_acc = rewards
                    best_choose=[[len(act[0]),len(act[1])],[test_acc],[num_index]]
                    best_actions=global_opt
                    best_vars=new_taskvars
                    best_accs=test_acc
                    np.savetxt('./choose/pmnist_action'+str(task_id)+'.txt', np.array(global_opt))  # the allocated neurons for all task
                    np.savetxt('./choose/pmnist_option'+str(task_id)+'.txt', np.array(options))  # the neurons for current task
                Areward_ = rewards  # *acc3#(acc_a)#*reward_)
                if iter==0:
                    reward=Areward_
                    #print(reward)
                else:
                    #print (reward_buffer)
                    reward = Areward_- ema(reward_buffer)#*0.49# * 0.9 #0.49
                #if there exist some extreme conditions we will re-initialize the controller.
                for ai in num_index:
                    if ai == 0:
                        reward = -1
                        sess2.run(init)
                        print ('no neurons!!!')
                # if  vali_acc<0.5:
                #     sess2.run(init)
                total_reward += reward
                ACC_regard.append(Areward_)
                print 'reward', reward, 'episode', iter, 'actions',[[len(act[0]),len(act[1])],[test_acc],[num_index]]
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
            #draw_pic_metric_ACC(result_acc)
        return best_vars,best_actions,best_accs
def basic_ops(train_sess,data):
    with tf.variable_scope('first_task_network'):
        NUM_NU = 1000
        max_layers = 2
        # define input andtarget placeholders
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        ACTION_SET = tf.placeholder(tf.int32, [2, NUM_NU])
        OLD_ACTION_SET = tf.placeholder(tf.int32, [2, NUM_NU])

        # variable initialization functions
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.08)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        W1 = weight_variable([784, NUM_NU])  # layer one
        b1 = bias_variable([NUM_NU])
        W2 = weight_variable([NUM_NU, NUM_NU])  # layer four
        b2 = bias_variable([NUM_NU])
        W3 = weight_variable([NUM_NU, 10])
        b3 = bias_variable([10])
        h1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # hidden layer 1
        h1s = tf.transpose(h1, [1, 0])
        h1s = tf.transpose((tf.gather(h1s, ACTION_SET[0])))  # operation on hidden layer
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
        for v in range(max_layers):
            if v == 0:  # first layer
                S = tf.transpose(W_u[v][0])
                new_W.append((tf.transpose(tf.gather(S, OLD_ACTION_SET[v])), W_u[v][1]))
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
    savers = tf.train.Saver()
    print("Training the first task!")
    vars_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #print('task network',vars_lists)
    init = tf.variables_initializer(vars_lists)
    train_sess.run(init)
    # action=np.loadtxt('./mnist_res/action1.txt')
    action1 = [1.0] * 312 + [0.0] * 688
    action2 = [1.0] * 128 + [0.0] * 872
    action_ = [action1, action2]
    actions = []
    real_action = []
    options = []
    for i in range(len(action_)):
        option = []
        opt = []
        for j in range(len(action_[i])):
            if action_[i][j] > 0.5:
                option.append(j)
                actions.append(1)
                opt.append(j)
            else:
                actions.append(0)
                option.append(NUM_NU)
        options.append(option)
        real_action.append(len(opt))
    #print 'real_option', real_action
    # print 'options', options
    neu_1 = []
    neu_2 = []
    neuset = []
    neu_1.append((real_action[0]))
    neu_2.append((real_action[1]))
    neuset.append(real_action)
    old_opt = options  # np.array([[1000]*1000,[1000]*1000]) # all can be used
    # print(old_opt)
    new_opt = options
    for step in range(9000):
        train_batch = data.train.next_batch(100)
        _, batch_loss = train_sess.run([train_step_, cross_entropy],
                                       feed_dict={x: train_batch[0], y_: train_batch[1], ACTION_SET: new_opt,
                                                  OLD_ACTION_SET: old_opt}
                                       )
        # acc_test = accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels,
        #                                     ACTION_SET: old_opt})  # task-A
        # print('validation accuracy', acc_test,step)
    acc_test = accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels,
                                        ACTION_SET: old_opt})  # task-A
    val = accuracy.eval(feed_dict={x: data.validation.images, y_: data.validation.labels,
                                   ACTION_SET: old_opt})  # task-A
    print('validation accuracy', val)
    print('test accuracy', acc_test)
    savers.save(train_sess, './vars/mnist_task1.pkt')
    np.savetxt('./choose/pmnist_action1.txt', np.array(options))
    np.savetxt('./choose/pmnist_option1.txt', np.array(options))
    return train_sess.run([W1, b1, W2, b2, W3, b3]),options,acc_test

def train():
    dataset=[mnist]
    for i in range(0,9):
        new_mnist=permute_mnist(mnist,seed=i)
        dataset.append(new_mnist)
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
        else:
            task_vars,old_action,best_acc=ops(task_vars, old_action,data, task_id)
            BEST_ACCS.append(best_acc)
            print('task best accuracy',task_id,best_acc)
    print('vag_acc',np.mean(BEST_ACCS))

#formulate the 10 tasks
#just test
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
train()