from pathlib import Path
from concurrent import futures
import logging
import time
import threading
import json
import os

import numpy as np
import cv2
import grpc
import pyrealsense2.pyrealsense2 as rs

from . import sh_pointlist_pb2
from . import sh_pointlist_pb2_grpc

#
# ==========================================================================
#

DTYPE = np.float16
PT_COUNT = 100000
KNOWN_SENSORS = None
XFORM_FN = Path(__file__).parent / "data/transform_data.json"
verts = {}

#
# ==========================================================================
#

def load_transformations( fn=XFORM_FN ):
    data = json.load( open(fn,"r") )
    for sid in data.keys():
        data[sid]["rotation"] = np.array( data[sid]["rotation"], dtype=DTYPE )
        data[sid]["translation"] = np.array( data[sid]["translation"], dtype=DTYPE )
    return data

#
# ==========================================================================
#

class Lidar():
    def __init__(self, serial_number=None ):

        self.serial_number = serial_number

        self.config = rs.config()
        self.pipeline = rs.pipeline()
        self.pipeline_wrapper = rs.pipeline_wrapper( self.pipeline )
        
        self.pc = rs.pointcloud()
#        self.decimate = rs.decimation_filter()
#        self.decimate.set_option( rs.option.filter_magnitude, 4 )  # downsample!

        if serial_number is not None:
            self.config.enable_device( serial_number )
            self.config.enable_stream( rs.stream.depth, 640,480, rs.format.z16, 30 )

        self.pipeline_profile = self.config.resolve( self.pipeline_wrapper )
        self.device = self.pipeline_profile.get_device()

        self.depth_sensor = self.device.first_depth_sensor()
        
        self.depth_sensor.set_option( rs.option.visual_preset, rs.l500_visual_preset.max_range )
        self.depth_sensor.set_option( rs.option.confidence_threshold, 1 )  # int; 0, 1, 2, or 3  (3 is the highest confidence)

        self.pipeline.start( self.config )
        
    def get_frames(self):
        while True:
            try:
                return self.pipeline.wait_for_frames()
            except Exception as e:
                print( "Error getting frames:", str(e) )
                pass

    def stop(self):
        self.pipeline.stop()


class MyServer(sh_pointlist_pb2_grpc.SmokyHumansServicer):

    def GetPoints(self, request, context):
        
        global verts
        
        try:
            myverts = np.vstack( verts.values() )
            pt_cnt = myverts.shape[0]

            if pt_cnt > PT_COUNT:
                inds = np.random.permutation( myverts.shape[0] )
                myverts = myverts[ inds[0:PT_COUNT], : ]
                
            # encode and ship all of the points
            return sh_pointlist_pb2.PointList( points=myverts.ravel() )
        
        except Exception as e:
            print("Error:", str(e) )
            return sh_pointlist_pb2.PointList( points=[] )

#
# ==========================================================================
#

def lidar_thread( serial_number="abc" ):
    global verts
        
    print( f"Starting LIDAR thread {serial_number}..." )
    lidar = Lidar( serial_number )

    try:
        while True:
            frames = lidar.get_frames()
            depth = frames.get_depth_frame()
            depth_vf = depth.as_video_frame()
            
            points = lidar.pc.calculate(depth_vf)
            _verts = np.asarray( points.get_vertices(2), dtype=DTYPE ).reshape(-1, 3)

            #
            # remove all invalid points
            #
            
            np_depth = np.asanyarray( depth_vf.get_data(), dtype=np.uint16 )
            orig_np_depth = np.copy( np_depth )
            
            np_depth = np.right_shift( np_depth, 8 ).astype( np.uint8) # convert from uint16 to uint8, keeping MSBs instead of LSBs.
            np_depth = cv2.medianBlur( np_depth, 7 ) # 3, 5, 7, 11
            np_depth = np_depth.astype( DTYPE )

            good_points = np.logical_and( np_depth > 0, orig_np_depth > 0 )
            
            _verts = _verts[ good_points.ravel(), : ]
            
            #
            # Transform points into unified coordinate system
            #

            _verts = np.dot( _verts, KNOWN_SENSORS[serial_number]["rotation"] )  + KNOWN_SENSORS[serial_number]["translation"]
            _verts[:,2] = -_verts[:,2]  # flip the z axis... don't know why our transform does this!

            #
            # Clip points outside of bounding box
            #
            
            good_points = np.logical_and(
                np.logical_and(
                    np.logical_and( _verts[:,0] > -3.5, _verts[:,0] < 3.5 ), 
                    np.logical_and( _verts[:,1] > -1, _verts[:,1] < 2.5 )
                    ),
                np.logical_and( _verts[:,2] > 0.05, _verts[:,2] < 10 ),
            )
            _verts = _verts[ good_points.ravel(), : ]

            verts[ serial_number ] = _verts

           
    finally:
        lidar.stop()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sh_pointlist_pb2_grpc.add_SmokyHumansServicer_to_server( MyServer(), server )
    server.add_insecure_port( '[::]:50051' )
    server.start()
    server.wait_for_termination()

def watch_transforms( name ):
    global KNOWN_SENSORS
    oldt = os.stat( XFORM_FN ).st_mtime
    while True:
        time.sleep(1.0)
        newt = os.stat( XFORM_FN ).st_mtime
        if newt != oldt:
            print("Reloading transform...")
            KNOWN_SENSORS = load_transformations( XFORM_FN )
            oldt = newt

#
# ==========================================================================
#

def main():
    global KNOWN_SENSORS
    
    KNOWN_SENSORS = load_transformations( XFORM_FN )
    
    x = threading.Thread( target=watch_transforms, args=(1,) )
    x.start()
    
    for serial_number in list(KNOWN_SENSORS.keys()):
        print( "FOUND SENSOR: ", serial_number )
    
    for serial_number in list(KNOWN_SENSORS.keys()):
        x = threading.Thread( target=lambda: lidar_thread(serial_number) )
        x.start()
        time.sleep( 0.5 * 1/30.0 )  # attempt to stagger starts to reduce noise (not sure if this works)

    logging.basicConfig()
    serve()

    
if __name__ == '__main__':
    main()
