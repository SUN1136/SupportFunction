import numpy as np
import binvox_rw
import tensorflow.compat.v1 as tf

with open("./model_normalized.solid.binvox", "rb") as f:
    model = binvox_rw.read_as_coord_array(f)

# points = []
# for i in range(128):
#     for j in range(128):
#         for k in range(128):
#             if model.data[i, j, k]:
#                 points.append(np.array([i, j, k]))

# points = np.array(points)
# points = points / 128 - 0.5
# points = points * model.scale - np.array([model.translate[1], model.translate[0], 0.0])

print(model.scale)
print(model.translate)
print(model.dims)
print(model.axis_order)

points = np.transpose(model.data)
points = points / 128 * model.scale + np.array(model.translate)

with tf.io.gfile.GFile("./solid.csv", "w") as fout:
    for i in range(points.shape[0]):
        fout.write("{0},{1},{2}\n".format(points[i, 0], points[i, 1], points[i, 2]))
