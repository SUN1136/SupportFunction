import numpy as np
import os
import tensorflow as tf



def off2obj(name):
    off_path = "./dataset/off/" + name + "/"
    off_list = os.listdir(off_path)
    off_num = len(off_list)

    if not tf.io.gfile.isdir("./dataset/objects/" + name):
        tf.io.gfile.makedirs("./dataset/objects/" + name)

    for i in range(off_num):
        off = off_list[i]
        obj = off.rstrip(".off")
        
        f = open(off_path + off)
        length = 0
        while "\n" in f.readline():
            length += 1
        f.close()

        head_idx = 1
        vert = []
        faces = []
        f = open(off_path + off)
        out = "# " + obj + "\n"
        for j in range(length):
            line = f.readline().split()

            if j == 0:
                if line[0] == "OFF":
                    head_idx = 1
                elif line[0] != "OFF" and ("OFF" in line[0]):
                    head_idx = 0
            
            if j > head_idx:
                y = [float(value) for value in line]
                if len(y) == 3:
                    vert.append(y)
                elif len(y) == 4:
                    faces.append(y[1:])
        
        vert = np.array(vert)
        max_vert = np.max(vert, axis = 0)
        min_vert = np.min(vert, axis = 0)
        cent_vert = (max_vert + min_vert) / 2
        vert = vert - cent_vert.reshape(1, 3)
        
        max_abs = np.max(np.abs(vert))
        scale = 0.4 / max_abs
        vert = vert * scale

        for j in range(vert.shape[0]):
            out += "v " + str(vert[j, 0]) + " " + str(vert[j, 1]) + " " + str(vert[j, 2]) + "\n"
        
        faces = np.array(faces)
        for j in range(faces.shape[0]):
            out += "f " + str(int(faces[j, 0]+1)) + " " + str(int(faces[j, 1]+1)) + " " + str(int(faces[j, 2]+1)) + "\n"
        
        w = open("./dataset/objects/" + name + "/" + obj + ".obj", "w")
        w.write(out)
        w.close()
        f.close()
        print("Done: " + obj)



def main():
    np.random.seed(45)
    tf.random.set_seed(45)
    
    off_list = os.listdir("./dataset/off/")
    for i in range(len(off_list)):
        off2obj(off_list[i])




if __name__ == "__main__":
    main()