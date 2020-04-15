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


def convert_sharded(input, output, label, converter):
    num_shards = 10
    output_filebase = "{}/sharded_{}.record".format(output, label)

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)

        for index, img in enumerate(input):
            # tf_example = build_tf_record(img.tobytes(), "{}_{}_img".format(index, label), label)
            tf_example = converter.convert(img)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


# convert('../../data/quick_draw/full_numpy_bitmap_face.npy', 'face')
# convert('../../data/quick_draw/full_numpy_bitmap_leg.npy', 'leg')
# convert_sharded('../../data/quick_draw/full_numpy_bitmap_face.npy', 'face')