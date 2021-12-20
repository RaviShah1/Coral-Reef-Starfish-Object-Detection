import os
from os.path import exists
import glob
import pandas as pd
import io
import json
import xml.etree.ElementTree as ET
import contextlib2
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from object_detection.dataset_tools import tf_record_creation_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="TensorFlow TFRecord Generator")
parser.add_argument("-c",
                    "--csv_path",
                    help="Path to the train.csv file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored.", type=str)
parser.add_argument("-t",
                    "--train",
                    help="True if this is a training dataset, false if it is a validation dataset.", type=str)
parser.add_argument("-s",
                    "--shards",
                    help="The number of shards for the dataset", type=int)
parser.add_argument("-f",
                    "--holdout_fold",
                    help="The fold to holdout.", type=int)

args = parser.parse_args()

def create_tf_example(data_df: pd.DataFrame, video_id: int, video_frame: int):
    """
    Create a tf.Example entry for a given training image.
    """
    full_path = os.path.join(args.image_dir, os.path.join(f'video_{video_id}', f'{video_frame}.jpg'))
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    height = image.size[1] # Image height
    width = image.size[0] # Image width
    filename = f'{video_id}:{video_frame}'.encode('utf8') # Unique id of the image.
    encoded_image_data = None # Encoded image bytes
    image_format = 'jpeg'.encode('utf8') # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    rows = data_df[(data_df.video_id == video_id) & (data_df.video_frame == video_frame)]
    for _, row in rows.iterrows():
        annotations = json.loads(row.annotations.replace("'", '"'))
        for annotation in annotations:
            xmins.append(annotation['x'] / width) 
            xmaxs.append((annotation['x'] + annotation['width']) / width) 
            ymins.append(annotation['y'] / height) 
            ymaxs.append((annotation['y'] + annotation['height']) / height) 

            classes_text.append('COTS'.encode('utf8'))
            classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

def create_labels_file():
    label_map_str = """
    item {
        id: 1
        name: 'COTS'
        }
                    """

    if exists('dataset/label_map.pbtxt') is False:
        with open('dataset/label_map.pbtxt', 'w') as f:
            f.write(label_map_str)
        print('Successfully created label_map.pbtxt file')

if __name__ == '__main__':

    # label file
    create_labels_file()
    #writer = tf.python_io.TFRecordWriter(args.output_path)
    
    # setup df
    data_df = pd.read_csv(args.csv_path)
    if args.train =='train':
        data_df = data_df[data_df.fold != args.holdout_fold].reset_index(drop=True)
    else:
        data_df = data_df[data_df.fold == args.holdout_fold].reset_index(drop=True)
    
    # make records
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, args.output_path, args.shards)
        
        for index, row in data_df.iterrows():
            tf_example = create_tf_example(data_df, row.video_id, row.video_frame)
            output_shard_index = index % args.shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
    #writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
