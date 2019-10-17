# -*- coding: utf-8 -*-
# 卷积神经网络的输入输出以及训练流程和全连接神经网络基本一致；唯一区别在于神经网络中相邻两层的连接方式。
# 以图像分类为例，卷积神经网络的输入是图像的原始像素，输出层的每一个节点代表了不同类别的可信度。
# 参数增多除了导致计算速度减慢，还很容易导致过拟合问题。卷积神经网络能有效减少神经网络中参数个数。
# 输入 卷积 池化 全连接 softmax层
# 经过卷积层处理过的节点矩阵会变得更深，池化层不会改变三维矩阵的深度，但是可以缩小矩阵的大小
# 池化层既可以加快计算速度也有防止过拟合问题的作用；但The All Convolutional Net论文中指出池化层对模型效果的影响不大
# 特征提取： 卷积 池化。 分类： 全连接层 softmax（得到当前样例属于不同种类的概率分布情况）
# zero-padding
import tensorflow as tf

# 卷积层： 矩阵中对应元素点积和
filter_weight = tf.get_variable("weight", [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
biases = tf.get_variable("biases", [16], initializer=tf.constant_initializer(0.1))
## tf.nn.conv2d：卷积层前向传播， 有四个参数，第一个参数为当前层的节点矩阵；第二个参数提供了卷积层的权重；第三个参数为不同维度上的步长：
# 一个长度为4的数组，但是第一维和最后一维的数字要求一定是1，这是因为卷积层的步长只对矩阵的长和宽有效，
# 最后一个参数是填充，“SAME”表示添加全0填充，使得卷积过后的矩阵和输入的矩阵大小相等（步长为1时），“VALID”表示不添加
# 比如 一个3X3的矩阵：[[1, -1, 0], [-1, 2, 1], [0, 2, -2]]，经过2X2的过滤器的卷积过程（Tensorflow实现全0填充优先填充右下方）：
# VALID: [[1, -1, 0], [-1, 2, 1], [0, 2, -2]]--》 [[A, B], [C, D]]
# SAME: [[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 2, 1], [0, 0, 2, -2]] --> [[A, B], [C, D]]
## tf.nn.bias_add: 给每个节点加上偏置项，给下一层神经网络2X2矩阵中的每个值都加上这个偏置项
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding="SAME")
bias = tf.nn.bias_add(conv, biases)
# 将计算结果通过RELU激活函数完成去线性化
actived_conv = tf.nn.relu(bias)

# 池化层：最大池化 or 平均池化
# tf.nn.max_pool 实现最大池化，ksize提供了过滤器的尺寸，strides提供步长信息，padding和卷积类似
pool = tf.nn.max_pool(actived_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")




