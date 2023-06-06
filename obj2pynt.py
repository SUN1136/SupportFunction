import numpy as np
from os import path
import tensorflow as tf

from pyntcloud import PyntCloud



def make_example(point):
    return tf.train.Example(features = tf.train.Features(feature = {
        "point_samples" : tf.train.Feature(float_list = tf.train.FloatList(value = point))
    }))

def write_tfrecord(points, filename):
    writer = tf.io.TFRecordWriter(filename)
    ex = make_example(points)
    writer.write(ex.SerializeToString())
    writer.close()

def get_points(object, n):
    path = "./dataset/objects/" + object + ".obj"
    obj = PyntCloud.from_file(path)
    points = obj.get_sample("mesh_random", n = n, rgb = False, normals = False)
    points = points.to_numpy().reshape(-1, )

    [v1, v2, v3] = obj.get_mesh_vertices(rgb = False, normals = False)
    vertices = np.concatenate([v1, v2, v3], axis = -1)

    with tf.io.gfile.GFile("./dataset/meshes/" + object + "/meshes.csv", "w") as fout:
        for i in range(vertices.shape[0]):
            for j in range(3):
                fout.write("{0},{1},{2}\n".format(vertices[i, 3*j], vertices[i, 3*j+1], vertices[i, 3*j+2]))

    dataset_path = "./dataset"
    if not tf.io.gfile.isdir(dataset_path):
        tf.io.gfile.makedirs(dataset_path)
    example_path = dataset_path + "/" + object + "-train-data.tfrecords"
    write_tfrecord(points, example_path)

    return points



def main():
    tf.random.set_seed(45)
    np.random.seed(45)

    num_points = 100000
    points = get_points("cone", num_points)
    print(points)



if __name__ == "__main__":
    main()