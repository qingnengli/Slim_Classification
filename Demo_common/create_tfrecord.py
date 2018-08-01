# coding=utf-8

import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/slim/')
from datasets import dataset_utils
import math
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#  根据list路径  把数据转化为TFRecord
def convert_dataset(list_dir, split_name, img_dir, output_dir, num_tfrecord=3):
    list_name = "dataset_" + split_name + ".txt"
    list_path = os.path.join(list_dir,list_name)
    fd = open(list_path)
    lines = [line.split() for line in fd]
    fd.close()
    num_per_shard = int(math.ceil(len(lines) / float(num_tfrecord)))
    with tf.Graph().as_default():
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        with tf.Session('') as sess:
            for shard_id in range(num_tfrecord):
                output_path = os.path.join(output_dir,'cifar10_{:03}-of-{:03}-{}.tfrecord'.format(
                                            shard_id+1, num_tfrecord, split_name))
                tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(lines))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image {}/{} to the shard {}'.format(
                        i + 1, len(lines), shard_id+1))
                    sys.stdout.flush()
                    image_data = tf.gfile.FastGFile(os.path.join(img_dir, lines[i][0]), 'rb').read()
                    image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
                    height, width = image.shape[0], image.shape[1]
                    example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, int(lines[i][1])) #btype object(b'jpg') btype.feature
                    tfrecord_writer.write(example.SerializeToString())
                tfrecord_writer.close()
    sys.stdout.write('\n') #输出不会换行
    sys.stdout.flush()


if __name__ == '__main__':
    img_dir = '/home/amax/SIAT/Slim_Classification/Data-Cifar10/image'
    list_dir = '/home/amax/SIAT/Slim_Classification/Demo_common/dataset/'
    convert_dataset(list_dir, 'train',img_dir,'dataset/train')
    convert_dataset(list_dir, 'test', img_dir, 'dataset/test')