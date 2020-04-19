import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import contextlib2
from object_detection.core import standard_fields
from object_detection.utils import dataset_util

from src.quickdraw.tfrecord_utils import open_sharded_output_tfrecords


def plot_img(img):
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()


def convert(input, label, converter):
    writer = tf.io.TFRecordWriter("{}.tfrecord".format(label))

    img_array = np.load(input)[:10]
    for idx, img in enumerate(img_array):
        tf_example = converter.convert((img.tobytes(),
                                        "{}_{}_img".format(idx, label),
                                        label)
                                       )
        writer.write(tf_example.SerializeToString())

    writer.close()


class ShardedTFRecordConverter:

    def __init__(self, output, label, converter):
        self.label = label
        self.num_shards = 10
        self.output_filebase = "{}/sharded_{}.record".format(output, label)
        self.tf_record_close_stack = contextlib2.ExitStack()
        self.output_tfrecords = open_sharded_output_tfrecords(
            self.tf_record_close_stack, self.output_filebase, self.num_shards)
        self.converter = converter

    def convert_sharded(self, input, index):
        tf_example = self.converter.convert(input)
        output_shard_index = index % self.num_shards
        self.output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

    def close(self):
        self.tf_record_close_stack.close()

# convert('../../data/quick_draw/full_numpy_bitmap_face.npy', 'face')
# convert('../../data/quick_draw/full_numpy_bitmap_leg.npy', 'leg')
# convert_sharded('../../data/quick_draw/full_numpy_bitmap_face.npy', 'face')