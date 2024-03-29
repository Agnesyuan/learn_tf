# -*- coding: utf-8 -*-
import tensorflow as tf

# 定义神经网络相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点个数
FC_SIZE = 512

# 定义卷积神经网络的前向传播过程，train用于区分训练过程和测试过程
# 增加dropout方法，进一步提高模型可靠性防止过拟合，dropout只在训练时使用
def inference(input_tensor, train, regularizer):
    # 第一层卷积
    with tf.variable_scope("layer1-conv1"):
        conv1_weight = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], 
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, [1, 1, 1, 1], padding='SAME')  # input_tensor:[100, 28, 28, 1] conv1_weight: [5, 5, 1, 32]; conv1 : [100, 28, 28, 32]
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))  # relu1: [100, 28, 28, 32]
    
    # 第二层 池化
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')    # pool1: [100, 14, 14, 32]
    
    # 第三层 卷积
    with tf.variable_scope("layer3-conv2"):
        conv2_weight = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], # [5, 5, 32, 64]
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weight, [1, 1, 1, 1], padding='SAME')   # conv2: [100, 14, 14, 64]
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    # 第四层 池化
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # pool2: [100, 7, 7, 64]

    # 将第四层池化层的输出转化为第五层全连接的输入格式，需要将7*7*64拉成一个向量
    pool_shape = pool2.get_shape().as_list()  # [100, 7, 7, 64]     其中100表示batch的大小
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]  # 3136

    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])  # [100, 3136]

    with tf.variable_scope("layer5-fc1"):
        fc1_weight = tf.get_variable("weight", [nodes, FC_SIZE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc1_weight))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_biases)  # [100, 3136] * [3136, FC_SIZE] = [batch, FC_SIZE] [100, 512]
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接层的前向传播过程，这一层的输入为长度时512的向量，输出为10的一维向量
    # 结果需要通过softmax层
    with tf.variable_scope("layer6-fc2"):
        fc2_weight = tf.get_variable("weight", [FC_SIZE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weight))

        fc2_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weight) + fc2_biases  # [batch, FC_SIZE] * [FC_SIZE, OUTPUT_NODE] = [batch, OUTPUT_NODE]   [100, 10]

    return logit

