import numpy as np
from os import path
import tensorflow as tf

def make_example(point):
    return tf.train.Example(features = tf.train.Features(feature = {
        "point_samples" : tf.train.Feature(float_list = tf.train.FloatList(value = point))
    }))

def write_tfrecord(points, filename):
    writer = tf.io.TFRecordWriter(filename)
    ex = make_example(points)
    writer.write(ex.SerializeToString())
    writer.close()

def read_example(datapath):
    dataset = tf.data.TFRecordDataset(datapath)
    for record in dataset.take(1):
        print(repr(record))

def main():
    tf.random.set_seed(45)
    np.random.seed(45)
    
    scale = 0.5
    num_points = 100000
    points = []
    for i in range(num_points):
        face = i % 6
        if face == 0 or face == 1:
            x = face - 0.5
            y = np.random.rand() - 0.5
            z = np.random.rand() - 0.5
        elif face == 2 or face == 3:
            x = np.random.rand() - 0.5
            y = face - 2.5
            z = np.random.rand() - 0.5
        elif face == 4 or face == 5:
            x = np.random.rand() - 0.5
            y = np.random.rand() - 0.5
            z = face - 4.5
        points.append(x)
        points.append(y)
        points.append(z)
    points = np.array(points, dtype = np.float32) * scale

    dataset_path = "./dataset"
    if not tf.io.gfile.isdir(dataset_path):
        tf.io.gfile.makedirs(dataset_path)
    example_path = path.join(dataset_path, "cube-train-data.tfrecords")
    write_tfrecord(points, example_path)

    # read_example(example_path)


if __name__ == "__main__":
    main()