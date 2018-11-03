# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
构建图表(build the Graph)
经过三阶段的模式函数操作：inference()， loss()，和training()
1.inference() —— 尽可能地构建好图表，满足促使神经网络向前反馈并做出预测的要求
2.loss() —— 往inference图表中添加生成损失(loss)所需要的操作(ops)
3.training() —— 往损失图表中添加计算并应用梯度(gradients)所需的操作
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.python.platform
import tensorflow as tf

# MNIST数据集有10个分类，代表0-9个数字
NUM_CLASSES = 10

# MNIST图像集像素为28*28
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

"""inference()函数会尽可能地构建图表，做到返回包含了预测结果(output prediction)的Tensor
Args:
  images: 输入与占位符（Inputs and Placeholders)
  hidden1_units: 第一个隐含层
  hidden2_units: 第二个隐含层
Returns:
  softmax_linear: 输出结果的logits Tensor
API备注：
  tf.truncated_normal() 根据所得到的均值和标准差，生成一个随机分布，初始化权重
  tf.nn.relu() 0/1阶跃函数
  tf.matmul() 矩阵相乘
  tf.Variable() 生成初始值变量，必须指定初始值
  tf.name_scope() 与tf.Variable()组合使用，更加方便管理参数命名
"""
def inference(images, hidden1_units, hidden2_units):

  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

"""计算预测lost
  Args:
    logits: Logits张量, float - [batch_size, NUM_CLASSES].
    labels: 标签张量, int32 - [batch_size].
  Returns:
    loss: 损失
  API备注:
    tf.expand_dims() 维度增加一维
    tf.sparse_to_dense() 将稀疏表示形式转换为稠密张量,转化为1-hot张量
    tf.stack() 矩阵拼接函数,在新的张量阶上拼接，产生的张量的阶数将会增加
    tf.concat() 将两个张量在某一个维度(axis)合并起来,产生的张量的阶数不会发生变化
    tf.nn.softmax_cross_entropy_with_logits_v2() softmax的输出向量和样本的实际标签的交叉熵的对应向量
                                                用来比较inference()函数与1-hot标签所输出的logits Tensor
  """
def loss(logits, labels):

  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat([indices, labels], 1)
  onehot_labels = tf.sparse_to_dense(
      concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                          labels=onehot_labels,
                                                          name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

"""
training
  training()函数添加了通过梯度下降(gradient descent)将损失最小化所需的操作
  创建优化器并应用梯度下降法迭代寻优
  操作函数放在会话sess运行
  Args:
    loss: 损失张量
    learning_rate: 学习速率
  Returns:
    train_op: 训练操作
    API备注：
    tf.summary.scalar() 对标量数据汇总和记录
    tf.train.GradientDescentOptimizer() 按照要求的学习率(固定值)应用梯度下降法
    optimizer.minimize() 使用minimize函数更新系统中的三角权重、增加全局步骤
"""
def training(loss, learning_rate):

  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


"""
评估
  预测标签并给出评估效果
  Args:
    logits: Logits张量
    labels: 预测标签
  Returns:
    正确预测标签张量
  API备注：
  tf.nn.in_top_k() 用于计算预测的结果和实际结果的是否相等,返回一个bool类型的张量
  tf.reduce_sum() 压缩求和，用于降维
"""
def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))