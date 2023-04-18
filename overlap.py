from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from os import path
import pandas as pd


def support_function(v, p, x, o):
    n_v = v.shape[0]

    z = np.zeros((n_v, ))
    for i in range(n_v):
        z[i] = np.max([np.dot(v[i], x), 0])
    
    maxz = np.max(z)
    if maxz != 0:
        z /= maxz

        sumz = np.sum(z**p)
        powsum = sumz**(1/p-1)
        powz1 = z**(p-1)
        
        s = np.matmul(np.transpose(v), powz1) * powsum
        s += o
        
        powz2 = z**(p-2)
        dsdx = np.diag(powz2) - np.matmul(powz1.reshape(-1, 1), powz1.reshape(1, -1)) / sumz
        dsdx = np.matmul(np.matmul(np.transpose(v), dsdx), v)
        dsdx = dsdx * (p-1) * powsum / maxz
    else:
        s = o
        dsdx = np.zeros((3, 3))

    return s, dsdx

def residual(obj, idx1, idx2, var):
    o1 = obj["o"][idx1]
    o2 = obj["o"][idx2]
    v1 = obj["v"][idx1]
    v2 = obj["v"][idx2]
    p1 = obj["p"][idx1]
    p2 = obj["p"][idx2]

    s1, dsdx1 = support_function(v1, p1, var[:3], o1)
    s2, dsdx2 = support_function(v2, p2, -var[:3], o2)

    f = np.zeros((4, ))
    f[:3] = var[3]*(s1 - s2) + (1 - var[3])*(o1 - o2)
    f[3] = np.sum(var[:3]**2) - 1

    jac = np.zeros((4, 4))
    jac[:3, 3] = (s1 - s2 - o1 + o2)
    jac[:3, :3] = var[3]*(dsdx1 + dsdx2)
    jac[3, :3] = 2*var[:3]
    jac[3, 3] = 0.0

    return f, jac

def dogleg(dN, dC, tr_radius):
    if np.linalg.norm(dN) < tr_radius:
        return dN
    elif np.linalg.norm(dC) > tr_radius:
        return tr_radius*dC/np.linalg.norm(dC)
    else:
        a = np.linalg.norm(dN - dC)**2
        b = np.dot(dC, (dN - dC))
        c = b*b - a*(np.linalg.norm(dC)**2 - tr_radius**2)
        tau = (-b + np.sqrt(c)) / a
        return dC + tau*(dN - dC)

def contact_compute(obj, idx1, idx2, n_ie, n_g):
    o1 = obj["o"][idx1]
    o2 = obj["o"][idx2]
    o_bar = o2 - o1
    v1 = obj["v"][idx1]
    v2 = obj["v"][idx2]
    p1 = obj["p"][idx1]
    p2 = obj["p"][idx2]

    u0 = np.array([1, 1, 1])
    
    small = 1e-3
    V_ie = np.array([[1, 0, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0, 0, 1]])
    V_ie = np.transpose(V_ie) * small
    u_ie = o_bar / np.linalg.norm(o_bar)

    W_ie = np.zeros((3, 3))
    iter = 0
    # while(True):
    #     iter += 1
    #     for i in range(4):
    #         singular = False
    #         W_ie[:, 0] = V_ie[:, 0] if i > 0 else V_ie[:, 1]
    #         W_ie[:, 1] = V_ie[:, 1] if i > 1 else V_ie[:, 2]
    #         W_ie[:, 2] = V_ie[:, 2] if i > 2 else V_ie[:, 3]
    #         try:
    #             Winv_ie = np.linalg.inv(W_ie)
    #         except:
    #             singular = True
    #             print(i)
    #             break
    #         c_ie = np.matmul(Winv_ie, o_bar)
    #         if np.min(c_ie) >= 0:
    #             break
        
    #     if not singular:
    #         u_ie = np.matmul(np.transpose(Winv_ie), u0)
    #         u_ie /= np.linalg.norm(u_ie)
    #     s1, dsdx1 = support_function(v1, p1, u_ie, o1)
    #     s2, dsdx2 = support_function(v2, p2, -u_ie, o2)
    #     V_ie[:3, :3] = W_ie
    #     V_ie[:, 3] = s1 - s2 + o_bar
    #     if iter >= n_ie or singular:
    #         break
    
    s1, dsdx1 = support_function(v1, p1, u_ie, o1)
    s2, dsdx2 = support_function(v2, p2, -u_ie, o2)
    var = np.zeros((4, ))
    var[:3] = u_ie / np.linalg.norm(u_ie)
    var[3] = np.sqrt(np.linalg.norm(o_bar) / np.linalg.norm(s1 - s2 + o_bar))

    res_list = []
    tr_radius = 10.0
    rho = 0
    iter = 0
    while(True):
        iter += 1
        f, jac = residual(obj, idx1, idx2, var)
        res = np.linalg.norm(f)
        res_list.append(res)
        if res < 1e-6 or iter >= n_g:
            break
        
        dN = -np.matmul(np.linalg.inv(jac), f)
        grad = np.matmul(np.transpose(jac), f)
        dC = - np.linalg.norm(grad)**2 / np.linalg.norm(np.matmul(jac, grad))**2 * grad
        dD = dogleg(dN, dC, tr_radius)

        f_next, jac_next = residual(obj, idx1, idx2, var + dD)
        act_red = (np.linalg.norm(f)**2 - np.linalg.norm(f_next)**2) / 2
        pred_red = -np.dot(grad, dD) - np.linalg.norm(np.matmul(jac, dD))**2 / 2

        if pred_red == 0:
            rho = 1e10
        else:
            rho = act_red / pred_red

        if rho < 0.05:
            tr_radius = 0.25 * np.linalg.norm(dD)
        elif rho > 0.9:
            tr_radius = np.max([tr_radius, 3*np.linalg.norm(dD)])
        
        if rho > 0.05:
            var = var + dD
    
    s1, dsdx1 = support_function(v1, p1, var[:3], o1)
    s2, dsdx2 = support_function(v2, p2, -var[:3], o2)
    normal = var[:3] / np.linalg.norm(var[:3])
    gap = np.linalg.norm(s1 - s2)
    if var[3] < 1:
        gap = -gap
    
    feature = {"s":[s1, s2], "n":normal, "g":gap, "res":res, "iter":iter, "sig":var[3], "nx":np.linalg.norm(var[:3])}
    return feature, res_list




data = pd.read_csv("/home/sun/바탕화면/UROP/SupportFunction/models/pointcloud/0418_2_cube/stats.csv")
data = data.to_numpy()

nc = 3
nv = 10
nd = 1742

p = data[nc*nd+1:nc*nd+1+nc, 0].astype(float)
v = data[nc*nd+2+nc:nc*nd+2+nc+nc*nv, 0:3].astype(float)
x = np.array([1, 1, 1])

o1 = np.mean(v[:10], axis = 0)
o2 = np.mean(v[10:20], axis = 0)
v1 = v[:10] - o1
v2 = v[10:20] - o2
p1 = p[0]
p2 = p[1]

o3 = np.mean(v[20:30], axis = 0)
v3 = v[20:30] - o3
p3 = p[2]
# s1, dsdx1 = support_function(v1, p1, x)
# s2, dsdx2 = support_function(v2, p2, x)

obj = {"o":[o1, o2, o3], "v":[v1, v2, v3], "p":[p1, p2, p3]}

feature, res_list = contact_compute(obj, 0, 1, 0, 20)
print("Object 0 1")
print(feature)
feature, res_list = contact_compute(obj, 0, 2, 0, 20)
print("Object 0 2")
print(feature)
feature, res_list = contact_compute(obj, 1, 2, 0, 20)
print("Object 1 2")
print(feature)
