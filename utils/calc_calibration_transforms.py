import argparse                                                                                                                     
import numpy as np
import numpy.linalg as la
import transform_3D
import json

from tqdm import tqdm

from sensor_def import *

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

def main():
    args = parse_args()
    
    sensors = locations[ args.location ]
    calibration_data = json.load( open(args.calibration_file) )

    transform_data = {}
    
    # Computer the transformations
    for sensor in sensors:
        sid = sensor["serial"]
        from_verts = np.asarray( calibration_data[sid]['from_verts'] )
        to_verts = np.asarray( calibration_data[sid]['to_verts'] )            

        print( "=====================================================" )
#        print( from_verts )
#        print( to_verts )
        
        rotation, translation = transform_3D.rigid_transform_3D(from_verts.T, to_verts.T)

        transform_data[ sid ] = {
            "rotation": rotation.T.tolist(),
            "translation": translation.T.tolist(),
        }
        
    with open( args.transform_file, "w") as file:
        json.dump(transform_data, file)

if __name__ == "__main__":
    main()
