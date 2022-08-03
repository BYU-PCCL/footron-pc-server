import pyrealsense2.pyrealsense2 as rs

from sensor_def import *

def reset_sensor( serial ):
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    config.enable_device( serial )
    pipeline_profile = config.resolve(pipeline_wrapper)
    
    device = pipeline_profile.get_device()
    device.hardware_reset()


def main():
    sensors = locations[ "wall" ]
    for sensor in sensors:
        sid = sensor['serial']
        print( f"Trying to reset {sid}..." )
        reset_sensor( sid )

if __name__ == "__main__":
    main()
