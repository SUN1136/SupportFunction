import numpy as np
import tensorflow as tf
import pandas as pd

def make_example(point, nearsurf, out):
    return tf.train.Example(features = tf.train.Features(feature = {
        "point_samples" : tf.train.Feature(float_list = tf.train.FloatList(value = point)), 
        "nearsurf_samples" : tf.train.Feature(float_list = tf.train.FloatList(value = nearsurf)), 
        "out_samples" : tf.train.Feature(float_list = tf.train.FloatList(value = out))
    }))

def write_tfrecord(points, nearsurf, out, filename):
    writer = tf.io.TFRecordWriter(filename)
    ex = make_example(points, nearsurf, out)
    writer.write(ex.SerializeToString())
    writer.close()



tf.random.set_seed(45)
np.random.seed(45)

pointsPath = "./data/points.txt"
points = pd.read_csv(pointsPath, sep = "\t", header = None)
points = points.to_numpy().reshape(-1, )
points = points[:-1]
points = points.reshape(-1, 3)

numPath = "./data/pointnum.txt"
pointNum = pd.read_csv(numPath, sep = "\t", header = None)
pointNum = pointNum.to_numpy().reshape(-1, )
pointNum = pointNum[:-1].astype(np.int32)

camPath = "./data/caminfo.txt"
caminfo = pd.read_csv(camPath, sep = "\t", header = None)
caminfo = caminfo.to_numpy().reshape(-1, )
caminfo = caminfo[:-1]
caminfo = caminfo.reshape(-1, 3)



pointAcc = np.copy(pointNum)
for i in range(pointAcc.shape[0] - 1):
    idx = pointAcc.shape[0] - 1 - i
    for j in range(idx):
        pointAcc[idx] = pointAcc[idx] + pointAcc[j]
pointAcc = pointAcc - pointNum

samples = []
nearsurf_samples = []
tmp_samples = []
out_samples = []
n = 100000
for i in range(n):
    vidx = np.random.choice(20, 1, False, pointNum/np.sum(pointNum))
    vidx = vidx[0]
    pidx = np.random.choice(pointNum[vidx], 1, False)
    pidx = pidx[0]

    pidx = pidx + pointAcc[vidx]
    surfp = points[pidx, :]
    samples.append(surfp)

    camp = caminfo[vidx]
    outp = surfp + (camp - surfp)/np.linalg.norm(camp - surfp) * (np.random.rand()*0.05+0.001)
    nearsurf_samples.append(outp)

    outp = surfp + (camp - surfp)/np.linalg.norm(camp - surfp) * (np.random.rand()*0.1+0.05)
    out_samples.append(outp)
    
    # mag = np.random.rand(3, )
    # for j in range(3):
    #     outp = surfp + (camp - surfp) * mag[j]
    #     if (np.abs(outp[0]) <= 1.0) and (np.abs(outp[1]) <= 1.0) and (np.abs(outp[2]) <= 1.0):
    #         tmp_samples.append(outp)

# tmp_samples = np.array(tmp_samples)
# out_idx = np.random.choice(tmp_samples.shape[0], n, False)
# for i in range(n):
#     out_samples.append(tmp_samples[out_idx[i], :])



samples = np.array(samples)
nearsurf_samples = np.array(nearsurf_samples)
out_samples = np.array(out_samples)

with tf.io.gfile.GFile("./data/points.csv", "w") as fout:
    for i in range(samples.shape[0]):
        fout.write("{0},{1},{2}\n".format(samples[i, 0], samples[i, 1], samples[i, 2]))

with tf.io.gfile.GFile("./data/nearsurf_points.csv", "w") as fout:
    for i in range(nearsurf_samples.shape[0]):
        fout.write("{0},{1},{2}\n".format(nearsurf_samples[i, 0], nearsurf_samples[i, 1], nearsurf_samples[i, 2]))

with tf.io.gfile.GFile("./data/out_points.csv", "w") as fout:
    for i in range(out_samples.shape[0]):
        fout.write("{0},{1},{2}\n".format(out_samples[i, 0], out_samples[i, 1], out_samples[i, 2]))


samples = samples.reshape(-1, )
nearsurf_samples = nearsurf_samples.reshape(-1, )
out_samples = out_samples.reshape(-1, )
example_path = "./data/airplane_0-train-data.tfrecords"
write_tfrecord(samples, nearsurf_samples, out_samples, example_path)