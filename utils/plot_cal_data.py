import argparse                                                                                                                     
import numpy as np
import json

import matplotlib.pyplot as plt

from tqdm import tqdm

from sensor_def import *

DTYPE=np.float16

#
# ============================================================
#

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration_file', type=str, default="./calibration_data.json")
    parser.add_argument('--transform_file', type=str, default="./transform_data.json")    
    parser.add_argument('--location', type=str, default="wall")
    parser.add_argument('--board', type=str, default="canvas")    

    return parser.parse_args()

def load_transformations( fn="transform_data.json" ):
    data = json.load( open(fn,"r") )
    for sid in data.keys():
        data[sid]["rotation"] = np.array( data[sid]["rotation"], dtype=DTYPE )
        data[sid]["translation"] = np.array( data[sid]["translation"], dtype=DTYPE )
    return data

def view_verts_3D(from_verts,to_verts,t_verts):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1,2,1,projection="3d")
    x, y, z = from_verts[:, 0], from_verts[:, 1], from_verts[:, 2]
    ax.scatter(x, y, z)

    ax = fig.add_subplot(1,2,2,projection="3d")
    
    x, y, z = to_verts[:, 0], to_verts[:, 1], to_verts[:, 2]
    ax.scatter(x, y, z)

    x, y, z = t_verts[:, 0], t_verts[:, 1], t_verts[:, 2]
    ax.scatter(x, y, z)

#    ax.axis('equal')
    ax.set_xlim([0,2])
    ax.set_ylim([0,2])
    ax.set_zlim([0,2])        
    
    plt.show()

def main():
    args = parse_args()

    transforms = load_transformations( args.transform_file )       
    
    sensors = locations[ args.location ]
    calibration_data = json.load( open(args.calibration_file) )

    for sensor in sensors:
        sid = sensor["serial"]
        from_verts = np.asarray( calibration_data[sid]['from_verts'] )
        to_verts = np.asarray( calibration_data[sid]['to_verts'] )            

        t_verts = np.dot( from_verts, transforms[sid]["rotation"] )  + transforms[sid]["translation"]
        
        view_verts_3D( from_verts, to_verts, t_verts )

        plt.show()
        plt.pause(0.1)
        
if __name__ == "__main__":
    main()
