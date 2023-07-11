import numpy as np
import tensorflow as tf
import pandas as pd

import open3d as o3d

from lib import convexdecoder



tf.random.set_seed(45)
np.random.seed(45)

path = "./dataset/objects/shapenet/airplane/airplane_0.obj"
mesh = o3d.io.read_triangle_mesh(path)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)


N = 50000
min_bound = mesh.vertex.positions.min(0).numpy()
max_bound = mesh.vertex.positions.max(0).numpy()

mean = (min_bound + max_bound) / 2
max_bound = (max_bound - mean)*1.2 + mean
min_bound = (min_bound - mean)*1.2 + mean

query_points = np.random.uniform(low = min_bound, high = max_bound, size = [N, 3]).astype(np.float32)

occ = scene.compute_occupancy(query_points)
occ = occ.numpy().astype(np.int32)



N_c = 20
N_v = 20

cvx_path = "./models/pointcloud/stats.csv"
cvx_data = pd.read_csv(cvx_path, header = None)
cvx_data = cvx_data.to_numpy()

smoothness = cvx_data[1:1+N_c, 0].astype(np.float32)
vertices = cvx_data[2+N_c:2+N_c+N_c*N_v, 0:3].astype(np.float32)
vertices = vertices.reshape(N_c, N_v, 3)

smoothness = np.expand_dims(smoothness, axis = 0)
vertices = np.expand_dims(vertices, axis = 0)

smoothness = tf.convert_to_tensor(smoothness)
vertices = tf.convert_to_tensor(vertices)

translations = tf.reduce_mean(vertices, axis = 2, keepdims = True)
vertices = vertices - translations

points = np.expand_dims(np.expand_dims(query_points, axis = 0), axis = 0)
points = tf.convert_to_tensor(points)
points = tf.tile(points, [1, N_c, 1, 1, ])


cvx = convexdecoder.ConvexDecoder(20, 21)
distances = cvx._compute_distance(vertices, smoothness, translations, points)

cvx_occ = tf.squeeze(tf.squeeze(tf.reduce_min(distances, axis = 1), axis = -1), axis = 0)
cvx_occ = tf.where(cvx_occ < 0, tf.ones_like(cvx_occ), tf.zeros_like(cvx_occ))
cvx_occ = cvx_occ.numpy().astype(np.int32)

union = 0
overlap = 0
for i in range(N):
    if occ[i] == 1 or cvx_occ[i] == 1:
        union += 1
        if occ[i] == 1 and cvx_occ[i] == 1:
            overlap += 1

iou = overlap / union
print("IoU: ", iou)

with tf.io.gfile.GFile("./occ.csv", "w") as fout:
    for i in range(N):
        fout.write("{0},{1},{2},{3},{4}\n".format(query_points[i, 0], query_points[i, 1], query_points[i, 2], occ[i], cvx_occ[i]))


