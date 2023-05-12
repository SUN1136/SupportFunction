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

def get_cube(num_points, scale, obj):
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
    example_path = path.join(dataset_path, obj + "-train-data.tfrecords")

    return points, example_path

def get_L(num_points, scale, obj):
    points = []
    for i in range(num_points):
        face = i % 14
        if face == 0 or face == 6:
            x = np.random.rand()*0.25 - 0.125
            y = np.random.rand()*0.25 - 0.25
            z = 0.25 if face == 0 else -0.25
        if face == 1 or face == 3:
            x = 0.125 if face == 1 else -0.125
            y = np.random.rand()*0.25 - 0.25
            z = np.random.rand()*0.25
        if face == 2 or face == 4:
            x = np.random.rand()*0.25 - 0.125
            y = 0 if face == 2 else -0.25
            z = np.random.rand()*0.25
        if face == 5 or face == 7:
            x = 0.125 if face == 5 else -0.125
            y = np.random.rand()*0.25 - 0.25
            z = np.random.rand()*0.25 - 0.25
        if face == 8 or face == 13:
            x = np.random.rand()*0.25 - 0.125
            y = -0.25 if face == 8 else 0.25
            z = np.random.rand()*0.25 - 0.25
        if face == 9 or face == 11:
            x = 0.125 if face == 9 else -0.125
            y = np.random.rand()*0.25
            z = np.random.rand()*0.25 - 0.25
        if face == 10 or face == 12:
            x = np.random.rand()*0.25 - 0.125
            y = np.random.rand()*0.25
            z = -0.25 if face == 10 else 0
        y = y - 0.1
        z = z - 0.1
        points.append(x)
        points.append(y)
        points.append(z)
    points = np.array(points, dtype = np.float32) * scale

    dataset_path = "./dataset"
    if not tf.io.gfile.isdir(dataset_path):
        tf.io.gfile.makedirs(dataset_path)
    example_path = path.join(dataset_path, obj + "-train-data.tfrecords")

    return points, example_path

def get_sphere(num_points, scale, obj):
    points = []

    n_th1 = 1000
    th1_list = []
    prob_list = []
    for i in range(1, n_th1):
        th1_list.append(i/n_th1 * np.pi - np.pi/2)
        prob_list.append(np.cos(i/n_th1 * np.pi - np.pi/2))
    th1_list = np.array(th1_list).reshape(-1, )
    prob_list = np.array(prob_list).reshape(-1, )
    prob_list = prob_list / np.sum(prob_list)

    for i in range(num_points):
        th1 = np.random.choice(th1_list, 1, p = prob_list)[0]
        th2 = np.random.rand() * 2*np.pi
        x = np.cos(th1) * np.cos(th2)
        y = np.cos(th1) * np.sin(th2)
        z = np.sin(th1)
        points.append(x)
        points.append(y)
        points.append(z)
    points = np.array(points, dtype = np.float32) * scale

    dataset_path = "./dataset"
    if not tf.io.gfile.isdir(dataset_path):
        tf.io.gfile.makedirs(dataset_path)
    example_path = path.join(dataset_path, obj + "-train-data.tfrecords")

    return points, example_path

def get_double_sphere(num_points, scale, obj):
    points = []

    n_th1 = 1000
    th1_list = []
    prob_list = []
    for i in range(1, n_th1):
        th1_list.append(i/n_th1 * np.pi - np.pi/2)
        prob_list.append(np.cos(i/n_th1 * np.pi - np.pi/2))
    th1_list = np.array(th1_list).reshape(-1, )
    prob_list = np.array(prob_list).reshape(-1, )
    prob_list = prob_list / np.sum(prob_list)

    for i in range(num_points):
        th1 = np.random.choice(th1_list, 1, p = prob_list)[0]
        th2 = np.random.rand() * 2*np.pi
        x = np.cos(th1) * np.cos(th2)
        y = np.cos(th1) * np.sin(th2)
        z = np.sin(th1)
        trans = 2 if i % 2 == 0 else -2
        x = x + trans
        y = y + trans
        z = z + trans
        points.append(x)
        points.append(y)
        points.append(z)
    points = np.array(points, dtype = np.float32) * scale

    dataset_path = "./dataset"
    if not tf.io.gfile.isdir(dataset_path):
        tf.io.gfile.makedirs(dataset_path)
    example_path = path.join(dataset_path, obj + "-train-data.tfrecords")

    return points, example_path

def get_cube_vertex(num_points, scale, obj):
    points = []
    for i in range(round(num_points * 0.9)):
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
    
    idx2vertex = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5]])
    for i in range(num_points - round(num_points * 0.9)):
        idx = i % 8
        vertex = idx2vertex[idx]
        points.append(vertex[0])
        points.append(vertex[1])
        points.append(vertex[2])
    points = np.array(points, dtype = np.float32) * scale

    dataset_path = "./dataset"
    if not tf.io.gfile.isdir(dataset_path):
        tf.io.gfile.makedirs(dataset_path)
    example_path = path.join(dataset_path, obj + "-train-data.tfrecords")

    return points, example_path

def get_cube_edge(num_points, scale, obj):
    idx2edge = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]])
    points = []
    for i in range(num_points):
        edge = i % 12
        if edge < 4:
            x = np.random.rand() - 0.5
            y = idx2edge[edge, 0]
            z = idx2edge[edge, 1]
        elif edge < 8:
            y = np.random.rand() - 0.5
            x = idx2edge[edge-4, 0]
            z = idx2edge[edge-4, 1]
        else:
            z = np.random.rand() - 0.5
            y = idx2edge[edge-8, 0]
            x = idx2edge[edge-8, 1]
        points.append(x)
        points.append(y)
        points.append(z)
    points = np.array(points, dtype = np.float32) * scale

    dataset_path = "./dataset"
    if not tf.io.gfile.isdir(dataset_path):
        tf.io.gfile.makedirs(dataset_path)
    example_path = path.join(dataset_path, obj + "-train-data.tfrecords")

    return points, example_path

def get_microphone(num_points, scale, obj):
    points = []
    
    face_num = np.array([0, 1, 2])
    face_prob = np.array([143.0, 106.0, 21.0])
    face_prob = face_prob / np.sum(face_prob)

    n_th1 = 1000
    th1_list = []
    th_prob = []
    for i in range(1, n_th1):
        th1_list.append(i/n_th1*(3*np.pi / 4))
        th_prob.append(np.sin(i/n_th1*(3*np.pi / 4)))
    th1_list = np.array(th1_list).reshape(-1, )
    th_prob = np.array(th_prob).reshape(-1, )
    th_prob = th_prob / np.sum(th_prob)

    r = 0.2
    r2 = 0.1 * np.sqrt(2)
    h = 0.3 + 0.2 - r2

    for i in range(num_points):
        face = np.random.choice(face_num, 1, p = face_prob)[0]

        if face == 0:
            th1 = np.random.choice(th1_list, 1, p = th_prob)[0]
            th2 = np.random.rand() * 2*np.pi
            x = r * np.sin(th1) * np.cos(th2)
            y = r * np.sin(th1) * np.sin(th2)
            z = r * np.cos(th1)
            z += r2
        elif face == 1:
            th3 = np.random.rand() * 2*np.pi
            x = r2 * np.cos(th3)
            y = r2 * np.sin(th3)
            z = -np.random.rand() * h
        elif face == 2:
            th4 = np.random.rand() * 2*np.pi
            rv = np.random.rand() * r2
            x = rv * np.cos(th4)
            y = rv * np.sin(th4)
            z = -h
        
        points.append(x)
        points.append(y)
        points.append(z)
    points = np.array(points, dtype = np.float32) * scale

    dataset_path = "./dataset"
    if not tf.io.gfile.isdir(dataset_path):
        tf.io.gfile.makedirs(dataset_path)
    example_path = path.join(dataset_path, obj + "-train-data.tfrecords")

    return points, example_path

def get_points(obj, num_points):
    func_dict = {"cube":[get_cube, 0.5], "L":[get_L, 1], "sphere":[get_sphere, 0.25], "double_sphere":[get_double_sphere, 0.125], 
                 "cube_vertex":[get_cube_vertex, 0.5], "cube_edge":[get_cube_edge, 0.5], "microphone":[get_microphone, 1]}
    return func_dict[obj][0](num_points, func_dict[obj][1], obj)

def main():
    tf.random.set_seed(45)
    np.random.seed(45)

    num_points = 100000
    points, example_path = get_points("microphone", num_points)

    write_tfrecord(points, example_path)

    # with tf.io.gfile.GFile(path.join("./dataset/sphere.csv"), "w") as fout:
    #       fout.write("x,y,z\n")
    #       points_out = points.reshape(-1, 3)
    #       for i in range(points_out.shape[0]):
    #         fout.write("{0},{1},{2}\n".format(points_out[i, 0], points_out[i, 1], points_out[i, 2]))

    # read_example(example_path)


if __name__ == "__main__":
    main()