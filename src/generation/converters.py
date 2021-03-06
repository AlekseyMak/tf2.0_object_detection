import json

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import contextlib2
from object_detection.core import standard_fields
from object_detection.utils import dataset_util

class TfRecordConverter:

    def convert(self, entry):
        return None


class NpyConverter(TfRecordConverter):

    def convert(self, entry):
        # (entry, image_id, label):
        image, id, label = entry
        img_feature_map = {
            standard_fields.TfExampleFields.object_bbox_ymin:
                dataset_util.float_list_feature([0]),
            standard_fields.TfExampleFields.object_bbox_xmin:
                dataset_util.float_list_feature([0]),
            standard_fields.TfExampleFields.object_bbox_ymax:
                dataset_util.float_list_feature([1]),
            standard_fields.TfExampleFields.object_bbox_xmax:
                dataset_util.float_list_feature([1]),
            standard_fields.TfExampleFields.object_class_text:
                dataset_util.bytes_list_feature([bytes(label, 'utf-8')]),
            standard_fields.TfExampleFields.object_class_label:
                dataset_util.int64_list_feature([1]),
            # standard_fields.TfExampleFields.filename:
            #     dataset_util.bytes_feature('{}.jpg'.format(image_id)),
            standard_fields.TfExampleFields.source_id:
                dataset_util.bytes_feature(bytes(id, 'utf-8')),
            standard_fields.TfExampleFields.image_encoded:
                dataset_util.bytes_feature(image),
        }

        img_features = tf.train.Features(feature=img_feature_map)
        return tf.train.Example(features=img_features)


class JsonConverter(TfRecordConverter):

    def convert(self, entry):
        img_feature_map = {
            standard_fields.TfExampleFields.object_bbox_ymin:
                dataset_util.float_list_feature([entry["bb_box_ymin"]]),
            standard_fields.TfExampleFields.object_bbox_xmin:
                dataset_util.float_list_feature([entry["bb_box_xmin"]]),
            standard_fields.TfExampleFields.object_bbox_ymax:
                dataset_util.float_list_feature([entry["bb_box_ymax"]]),
            standard_fields.TfExampleFields.object_bbox_xmax:
                dataset_util.float_list_feature([entry["bb_box_xmax"]]),
            standard_fields.TfExampleFields.object_class_text:
                dataset_util.bytes_list_feature([bytes(entry["category"], 'utf-8')]),
            standard_fields.TfExampleFields.object_class_label:
                dataset_util.int64_list_feature([1]),
            # standard_fields.TfExampleFields.filename:
            #     dataset_util.bytes_feature('{}.jpg'.format(image_id)),
            standard_fields.TfExampleFields.source_id:
                dataset_util.bytes_feature(bytes(entry["id"], 'utf-8')),
            standard_fields.TfExampleFields.image_encoded:
                dataset_util.bytes_feature(entry['img']),
        }

        img_features = tf.train.Features(feature=img_feature_map)
        return tf.train.Example(features=img_features)

