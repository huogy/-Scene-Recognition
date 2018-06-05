import os
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import numpy as np

"""
# 将自己的数据转化为Tensorflow中TFrecords形式
def conver_to_tfrecords():
    cwd='/home/ricky/pycharm-2017.3/bin/photos/'
    # os.walk()
    classes={'1_E','1_N','1_S','1_W','2_E',
            '2_N','2_S','2_W','3_E','3_N',
            '3_S','3_W','4_E','4_S','4_W',
            '5_E','5_N','5_S','5_W','6_E',
            '6_N','6_W','7_E','7_N','7_S',
            '7_W','8_N','8_S','8_W','9_W',
            '10_N','10_S','11_E', '11_N', '11_S',
            '11_W','12_E', '12_N', '12_S', '12_W',
            '13_N', '13_S', '13_W', '14_N', '14_S',
            '15_N', '15_W','16_S', '16_W','17_E',
            '17_N','18_E',  '18_S','19_E', '19_N',
            '19_S', '19_W','20_E', '20_N', '21_E',
            '21_N', '21_S','21_W','22_E', '22_N',
            '22_S', '22_W', '23_E', '23_N', '23_S',
            '23_W', '24_E', '24_N', '24_S', '24_W',
            '25_N', '25_S', '25_W', '26_N', '26_S',
            '26_W', '27_N', '27_S', '27_W','28_S',
            '29_E',  '29_S',  '30_S', '30_W','31_E',
            '31_N', '31_S', '32_E', '32_S', '32_W'}
    writer = tf.python_io.TFRecordWriter('school.records')
    for index,name in enumerate(classes):
        class_path=cwd+name+'/'
        for img_name in os.listdir(class_path):
            img_path=class_path+img_name
            img=Image.open(img_path)
            img= img.resize((227,227))#重新裁剪图片尺寸
            img_raw=img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()

conver_to_tfrecords()
"""
#从TFrecords读取数据
def read_and_decode(filename_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#return image and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])  #reshape image to 227
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #throw img tensor
    img = tf.cast(img, tf.float32) /  127.5 -1.#转化成img tensor
    label = tf.cast(features['label'], tf.int32) #转成 label tensor
    return img, label


#生成器  生成batch，来供给网络
def next_batch(train,batch_size,num_epochs):
    dir = '/home/huo/PycharmProjects/alexnet/11/school.records'

    filename_queue = tf.train.string_input_producer([dir],num_epochs=num_epochs)#队列
    img, label = read_and_decode(filename_queue)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    num_threads=2,#线程
                                                    capacity=2000,#队列最大长度
                                                    min_after_dequeue=500)

    label_batch = tf.one_hot(label_batch, depth=95)#转化为ont-hot格式

    return img_batch, label_batch

x = tf.placeholder(tf.float32, [None, 227, 227,3], name='input111')
y = tf.placeholder(tf.float32,[None,95],name='output')
BATCH_SIZE = 1
def inference(images, dropout):

    # conv1
    with tf.variable_scope('layer1-conv1'):
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias)

    # lrn1
    with tf.name_scope("layer2-pool1"):
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)
        # pool1
        pool1 = tf.nn.max_pool(lrn1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')


    # conv2
    with tf.variable_scope('layer1-conv2'):
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias)



    # lrn1
    with tf.name_scope("layer2-pool2"):
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)
        # pool1
        pool2 = tf.nn.max_pool(lrn2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')


    # conv3 13*13*256
    with tf.variable_scope('layer3-conv3'):
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias)



    # conv4  13*13*384
    with tf.variable_scope('layer4-conv4'):
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias)



    # conv5 13*13*384
    with tf.variable_scope('layer5-conv5'):
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias)

        # pool5
        pool5 = tf.nn.max_pool(conv5,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')

     #输出8*8*256
    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable("weight", [16384, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc = tf.reshape(pool5, [-1, fc1_weights.get_shape().as_list()[0]])
        fc1_biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        fc1 = tf.add(tf.matmul(fc, fc1_weights), fc1_biases)
        fc1 = tf.nn.relu(fc1)

    """
    # with tf.variable_scope('layer7-fc2'):
    #     fc2_weights = tf.get_variable("weight", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     fc2_biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
    #     fc2 = tf.add(tf.matmul(fc1, fc2_weights), fc2_biases)
    #     fc2 = tf.nn.relu(fc2)
    #     dropout
        # fc2 = tf.nn.dropout(fc2, dropout)
    """
    with tf.variable_scope('layer8-out'):
        out_weights = tf.get_variable("weight", [4096, 95], initializer=tf.truncated_normal_initializer(stddev=0.1))
        out_biases = tf.Variable(tf.constant(0.0, shape=[95], dtype=tf.float32), trainable=True, name='biases')
        out = tf.add(tf.matmul(fc1, out_weights), out_biases,name="outpr")
        return out


learning_rate = 1e-4
dropout = 0.5#dropout
MODEL_SAVE_PATH = "./alexnet/"
MODEL_NAME = "alexnet_model.ckpt"




#x_,y = next_batch(train=True,batch_size=32,num_epochs=100000)#将图片输入给网络
# pred = inference(x_, dropout)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))#定义cost值
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)#使用adam优化
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#计算准确率

image_file_name = sys.argv[0]
image_raw = tf.gfile.FastGFile('D:/pythonpycharm/11/21.jpg', 'rb').read()# bytes D:/pycharm/pycharmProject/11/1.png
image = tf.image.decode_png(image_raw)
image = tf.reshape(image, [ -1,227, 227, 3])
image = tf.cast(image, tf.float32)
# img = tf.decode_raw(image,tf.float32)  # Tensor
print(image.eval(session=tf.Session()))
logits = inference(image, 1)
predict0 = tf.argmax(logits, 1)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  # 初始化
with tf.Session()as sess:  # 会话 开始训练
    sess.run(init_op)
    saver = tf.train.Saver()
    saver.restore(sess, "D:/pythonpycharm/11/alexnet/alexnet_model.ckpt")
    print("Model restored.")
    image_raw = sess.run(image)
    label =sess.run(predict0,feed_dict={x:image_raw})
    print(label)
    sess.close()
    #start input enqueue threads
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    # try:
    #     step=0
    #     while not coord.should_stop():
    #         sess.run([x_,y])
    #         # import matplotlib.pyplot as plt
    #         # print(ss[1][0])
    #         # plt.imshow(255-ss[0][0])
    #         # plt.show()
    #         # print(ss)
    #         sess.run(optimizer)
    #         loss,acc=sess.run([cost,accuracy])
    #         if step%100 == 0:
    #             print("step= ",step,"---","loss=",loss,'---',"acc=",acc)
    #         step+=1
    #         #changed
    #         if acc>0.95 and step>85000 and step%1000==0:
    #             #graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["outpr"])
    #             tf.train.write_graph(sess.graph_def,"./alexnet/","alexnet_model.pb",as_text=False)
    #             saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=step)
    #             # frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, map(canonical_name, out_tensors))
    #             #tflite_model = tf.contrib.lite.toco_convert(graph, [img], [outpr])
    #             # global step = i
    # except tf.errors.OutOfRangeError:
    #     print('out of range')
    # finally:
    #     coord.request_stop()
    #     coord.join(threads)





