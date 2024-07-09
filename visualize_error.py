# Reference: https://matplotlib.org/stable/gallery/mplot3d/surface3d.html#sphx-glr-gallery-mplot3d-surface3d-py
# Reference https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html#sphx-glr-gallery-mplot3d-subplot3d-py

import matplotlib.pyplot as plt
import numpy
import sys
from utils.shared import *

from matplotlib import cm
import os

def fetch_list_patch_file(arg):
    path_dir = arg
    if not os.path.isdir(path_dir):
        path_dir = os.path.join(get_path_dir_output(arg), "errors")
    if not os.path.isdir(path_dir):
        return None
    list_name_file = sorted(list(os.listdir(path_dir)))
    assert len(list_name_file) > 1
    assert list_name_file[0] == "00000.npz"
    return list(map(lambda name_file: os.path.join(path_dir, name_file), list_name_file))

def load_file(list_path_file):
    list_npz = list(map(numpy.load, list_path_file))
    list_list_mse_sum = numpy.array(list(map(lambda npz: npz["tlist_mse_sum"], list_npz)))
    list_list_count = numpy.array(list(map(lambda npz: npz["tlist_count"], list_npz)))

    # Skip t=0 and epoch=0
    list_list_mse_sum = list_list_mse_sum[1:,1:]
    list_list_count = list_list_count[1:,1:]

    # Post process
    data = list_list_mse_sum / list_list_count
    data = numpy.log2(data)

    return data

def plot(data):
    assert len(data.shape) == 2

    xs = numpy.arange(data.shape[1]) + 1
    ys = numpy.arange(data.shape[0]) + 1
    xs, ys = numpy.meshgrid(xs, ys)
    zs = data

    # print(xs.shape)
    # print(ys.shape)
    # print(zs.shape)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    ax.set_xlabel("t")
    ax.set_ylabel("epoch")
    ax.set_zlabel("log2mse")

    # surf = ax.plot_wireframe(xs, ys, zs, rstride=1, cstride=1)
    surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

if __name__ == "__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) != 2:
        print("python visualize.py PATH_DIR or RUN_NAME")
    else:
        plot(load_file(fetch_list_patch_file(sys.argv[1])))