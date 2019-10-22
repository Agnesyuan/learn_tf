# -*- coding: utf-8 -*-
import glob
import os
import random
import numpy as np
import tensorflow as tf

# inception-v3瓶颈层的节点个数
BOTTLENECT_TENSOR_SIZE = 2048
# 在谷歌提供的inception-v3模型中，瓶颈层结果的张量名称为“pool_3/_reshape:0”
BOTTLENECT_TENSOR_NAME = "pool_3/_reshape:0"
# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = "DecodeJpeg/contents:0"
# 下载的模型文件目录
MODEL_DIR = "./inception_dec_2015"
# 下载训练好的模型文件名
MODEL_FILE = "tensorflow_inception_graph.pb"
# 将原始图像通过inception-v3模型计算得到的特征向量保存在bottleneck文件中
CACHE_DIR = "./bottleneck_test"
# 图片数据文件夹 子文件为类别
INPUT_DATA = "./flower_photos"
VALIDATION_PRECENTAGE = 10
TEST_PRECENTAGE = 10
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# 从数据文件夹中读取所有的图片列表并按训练、验证、测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)] # ./flower_photos, ./flower_photos/daisy
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ["jpg", "jpeg", "JPG", "JPEG"]
        file_list = []   # ['./flower/photos/roses/xx.jpg', '...']
        dir_name = os.path.basename(sub_dir) # roses, daisy 
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*."+extension)
            file_list.extend((glob.glob(file_glob)))  # glob.glob()返回所有路径下的符合条件的文件名列表
        if not file_list:
            continue

        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)  # ['xx.jpg', '...']
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {"dir": dir_name, "training": training_images,
                              "testing": testing_images, "validation":validation_images}
        # result = {'roses': {'dir': roses, 'training': ['xx.jpg', ''], 'testing': [], 'validation': [..]}, 
        #            'daisy':{...}}  一共五类图片 
    return result


# 通过类别名称、所属数据集和图片编号获取一张图片的地址
# image_lists为所有图片信息，image_dir根目录，label_name为类别名称，index为图片编号，category指定图片在哪个训练集
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists["dir"]
    full_path = os.path.join(image_dir, sub_dir, base_name)  # ./bottleneck/sunflowers/9056495873_66e351b17c_n.jpg
    return full_path

def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + ".txt"

#使用加载的训练好的网络处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 将当前图片作为输入，计算瓶颈张量的值
    # 这个张量的值就是这张图片的新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})  # shape : (1, 2048)
    # 经过卷积神经网络处理的结果是一个四维数据，需要将这个结果压缩成一个一维变量
    bottleneck_values = np.squeeze(bottleneck_values) # 从数组的形状中删除单维条目  # shape: (2048,)
    return bottleneck_values

# 获取一张图片经过inception-v3模型处理后的特征向量；先寻找已经计算并且保存的向量，若找不到则计算然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists["dir"]
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = tf.gfile.GFile(image_path, "rb").read()
        
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor) # (2048,)
        bottleneck_string = ",".join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, "w") as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, "r") as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(",")]
    return bottleneck_values     

# 随机选取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

# 获取全部的测试数据，在最终测试的时候在所有测试数据上计算正确率
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = "testing"
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
        return bottlenecks, ground_truths

def main(_):
    image_lists = create_image_lists(TEST_PRECENTAGE, VALIDATION_PRECENTAGE)
    n_classes = len(image_lists.keys())
    print(image_lists.keys())
    with tf.gfile.GFile(os.path.join(MODEL_DIR, MODEL_FILE), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECT_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
        bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECT_TENSOR_SIZE], name="BottleneckInputPlaceholder")
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name="GroundTruthInput")

        with tf.name_scope("final_training_ops"):
            weights = tf.get_variable("weights", [BOTTLENECT_TENSOR_SIZE, n_classes], initializer=tf.truncated_normal_initializer(stddev=0.001))
            #weights = tf.Variable(tf.truncated_normal_initializer([BOTTLENECT_TENSOR_SIZE, n_classes], stddev=0.001))
            biases = tf.Variable(tf.zeros([n_classes]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)

        # 定义交叉熵损失函数
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

        # 计算正确率
        with tf.name_scope("evalution"):
            correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            evalution_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # 训练过程
            for i in range(STEPS):
                train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH, "training", jpeg_data_tensor, bottleneck_tensor)
                sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

                if i % 100 == 0 or i+1 == STEPS:
                    validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                        sess, n_classes, image_lists, BATCH, "validation", jpeg_data_tensor, bottleneck_tensor)

                    validation_accuracy = sess.run(evalution_step,
                                                   feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})

                    print("Step %d: Validation accuracy on random sampled %d examples = %.1f%%" %
                          (i, BATCH, validation_accuracy*100))

            test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes,
                                                                       jpeg_data_tensor, bottleneck_tensor)

            test_accuracy = sess.run(evalution_step, feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
            print("Final test accuracy = %.1f%%" % (test_accuracy*100))

if __name__ == "__main__":
    tf.app.run()
















