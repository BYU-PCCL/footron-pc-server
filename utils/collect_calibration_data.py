import argparse                                                                                                                     
import pyrealsense2.pyrealsense2 as rs
from sklearn.linear_model import RANSACRegressor
import numpy as np
import numpy.linalg as la
import cv2

import matplotlib.pyplot as plt
import json

from tqdm import tqdm

from sensor_def import *

#
# ============================================================
#

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true') # default is false
    parser.add_argument('--clear', action='store_true') # default is false
    
    parser.add_argument('--zofs', type=float, default=0.0)
    parser.add_argument('--frame_count', type=int, default=500)
    
    parser.add_argument('--calibration_file', type=str, default="./calibration_data.json")

    parser.add_argument('--location', type=str, default="wall")
    parser.add_argument('--board', type=str, default="canvas")    

    return parser.parse_args()

#
# ============================================================
#

def get_inlier_mask(pc_verts):
    # Robustly fit linear model with RANSAC algorithm
    X, y = pc_verts[:,[0, 1]], pc_verts[:,[2]]
    # Maximum residual for a data sample to be classified as an inlier, good results
    # between 0.005 - 0.01
    ransac = RANSACRegressor(residual_threshold=0.005)
    ransac.fit(X, y)
    return ransac.inlier_mask_


def get_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    if serial is not None:
        config.enable_device(serial)
    pipeline_profile = config.resolve(pipeline_wrapper)

    # Enable highest resolution streams for more precise measurement and transformation
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

    device = pipeline_profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_sensor.set_option( rs.option.visual_preset, rs.l500_visual_preset.max_range )
    #depth_sensor.set_option( rs.option.confidence_threshold, 1 )  # int; 0, 1, 2, or 3
    
    # In case of issues, call hardware_reset() on device
    pipeline.start(config)

    return pipeline


def get_frames(pipeline):
    try:
        return pipeline.wait_for_frames()
    except RuntimeError as e:
        print(f"{e}; calling hardware_reset() on device")
        try:
            pipeline.get_active_profile().get_device().hardware_reset()
            return pipeline.wait_for_frames()
        except RuntimeError as e:
            print(f"{e}; even after hardware_reset()")
            raise e


def normalize(x):
    tmp = x - np.min(x)
    tmp /= np.max(tmp)
    return tmp


def chessboard_verts(image, board, crop=None):
    copy = image.copy()
    if crop is not None:
        (x1, x2), (y1, y2) = crop
        copy = copy[y1:y2, x1:x2]
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # Find the chess board corners
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

    # plt.imshow(gray)
    # plt.savefig("crop")

    ret, corners = cv2.findChessboardCorners(gray, (board["y_verts"], board["x_verts"]), None)

    # Throw error if we can't find board
    if not ret:
        raise Exception("Couldn't find chessboard verts")

    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)[:, 0, :]
    if crop is not None:
        return corners2 + np.array([x1, y1])
    return corners2


def crop_near(image, vert, radius=100):
    x, y = vert
    x1 = round(x - radius)
    x2 = round(x + radius)
    y1 = round(y - radius)
    y2 = round(y + radius)
    # We don't need to clip to avoid going out of index, np does it automatically
    cropped = image[y1:y2, x1:x2]
    return cropped


def color_near(color_image, center, radius, view=False):
    m, n = color_image.shape[:2]
    x, y = np.arange(n), np.arange(m)
    cx, cy = center
    circle_mask = (((x[np.newaxis, :] - cx) ** 2) + ((y[:, np.newaxis] - cy) ** 2)) < (
        radius**2
    )
    if view:
        copy = color_image.copy()
        copy[circle_mask] = [0, 0, 0]
        plt.imshow(copy)
        plt.show()
    color_circle = color_image[circle_mask]
    average_color = color_circle.mean(axis=0)
    return average_color


def color_center(vert, x_step, y_step):
    x_scale = 1.0
    y_scale = 1.1
    center = vert + (x_step * x_scale) + (y_step * y_scale)
    return center


def circle_color(color_image, vert, next_x, next_y):
    x_step = vert - next_x
    y_step = vert - next_y
    center = color_center(vert, x_step, y_step)
    step_radius_scale = 0.4
    radius = np.min([la.norm(x_step), la.norm(y_step)]) * step_radius_scale
    color = color_near(color_image, center, radius)
    return color


def verts_align(color_image, board_verts, board):
    y_verts = board["y_verts"]

    first, next_x_first, next_y_first = board_verts[[0, 1, y_verts]]
    last, next_x_last, next_y_last = board_verts[[-1, -2, -y_verts - 1]]

    first_color = circle_color(color_image, first, next_x_first, next_y_first)
    last_color = circle_color(color_image, last, next_x_last, next_y_last)

    red = np.array([255, 0, 0])
    blue = np.array([0, 0, 255])
    first_more_red = np.linalg.norm(red - first_color) < np.linalg.norm(
        red - last_color
    )
    last_more_blue = np.linalg.norm(blue - last_color) < np.linalg.norm(
        blue - first_color
    )

    # Both are flipped
    if not first_more_red and not last_more_blue:
        return board_verts[::-1]

    # One of them is flipped
    if not first_more_red or not last_more_blue:
        raise Exception("Orientation uncertain for board verticies")

    return board_verts


def aligned_board_verts(image, board, crop):
    board_verts = chessboard_verts(image, board, crop)
    aligned_board_verts = verts_align(image, board_verts, board)
    return aligned_board_verts


def chessboard_3D_verts(board,zofs):
    row_verts, col_verts, grid_size_in_meters = (
        board["x_verts"],
        board["y_verts"],
        board["square_size"],
    )
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    verts = np.zeros((row_verts * col_verts, 3), np.float32)
    verts[:, :2] = np.mgrid[0:col_verts, 0:row_verts].T.reshape(-1, 2)
    verts *= grid_size_in_meters
    verts[:,2] = zofs
    return verts


def view_circle(vert, next_x, next_y, ax):
    x_step = vert - next_x
    y_step = vert - next_y
    center = color_center(vert, x_step, y_step)
    step_radius_scale = 0.4
    radius = np.min([la.norm(x_step), la.norm(y_step)]) * step_radius_scale
    ax.add_patch(
        plt.Circle(
            center, radius * step_radius_scale, color="yellowgreen", fill=False, lw=2
        )
    )


def view_board_alignment(image, board, crop=None):
    plt.imshow(image)
    plt.show()
    plt.savefig("img")
    try:
        board_verts = chessboard_verts(image, board, crop)
    except:
        print("Could not find chessboard corners")
        return
    board_verts = verts_align(image, board_verts, board)
    y_verts = board["y_verts"]
    first, next_x_first, next_y_first = board_verts[[0, 1, y_verts]]
    last, next_x_last, next_y_last = board_verts[[-1, -2, -y_verts - 1]]
    fig, ax = plt.subplots()
    plt.imshow(image)
    for i, vert in enumerate(board_verts):
        plt.scatter([vert[0]], [vert[1]], s=15, c="r")
        plt.text(*vert, str(i))
    plt.colorbar()
    view_circle(first, next_x_first, next_y_first, ax)
    view_circle(last, next_x_last, next_y_last, ax)
    plt.savefig("board-align")


def run_view_board_alignment(sensor, board):
    pipeline = get_pipeline(sensor["serial"])
    # Let 3 frames go by before using them (the first couple are often blank)
    for _ in range(3):
        frames = get_frames(pipeline)
    frames = get_frames(pipeline)
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    view_board_alignment(color_image, board, sensor["crop"])


def view_verts_2D(depth_image, board_verts):
    plt.imshow(depth_image)
    for i, vert in enumerate(board_verts):
        plt.scatter([vert[0]], [vert[1]], s=15, c="r")
        print(depth_image[round(vert[1]), round(vert[0])])
        plt.text(*vert, str(i))
    plt.colorbar()
    plt.show()


def view_verts_3D(verts):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    ax.scatter(x, y, z)
    plt.show()


def amalgamate(points_by_frame, remove_zeros=True, median=False):
    method = np.median if median else np.mean
    if not remove_zeros:
        return method(points_by_frame, axis=0)
    points_by_point = np.stack(points_by_frame, axis=1)
    non_zero_mask = np.logical_and.reduce(points_by_point != 0, axis=2)

    # We will throw out any points that we're always zero
    valid_points = np.array([np.any(mask) for mask in non_zero_mask])

    # If all points are zero, keep one of them to maintain shape
    for mask in non_zero_mask:
        if np.all(~mask):
            mask[0] = True

    amalgamations = np.array(
        [method(x[m], axis=0) for x, m in zip(points_by_point, non_zero_mask)]
    )
    return amalgamations[valid_points], valid_points


def collect_data(sensor, board, frame_count=150, view_from_verts=True, debug=False, zofs=0.0):
    serial = sensor["serial"]
    pipeline = get_pipeline(serial)

    # Processing block
    pc = rs.pointcloud()

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Let 3 frames go by before using them (the first couple are often blank)
    for i in range(3):
        frames = pipeline.wait_for_frames()

    # Collection of verts over multiple frames to check mean and var
    board_verts_col = []
    from_verts_col = []

    # Display verticies in decimal rather than scientific
    np.set_printoptions(suppress=True)

    for i in tqdm(range(frame_count)):
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            crop = sensor["crop"]
            board_verts = aligned_board_verts(color_image, board, crop)
            points = pc.calculate(depth_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()

            verts = np.asanyarray(v).view(np.float32).reshape(1080, 1920, 3)

            if debug:
                # view_verts_2D(depth_image, board_verts)

                for vert in board_verts:
                    if depth_image[round(vert[1]), round(vert[0])] == 0:
                        print("couldn't compute depth at frame", i)

            from_verts = np.array(
                [verts[round(vert[1]), round(vert[0])] for vert in board_verts]
            )

            board_verts_col.append(board_verts)
            from_verts_col.append(from_verts)

        except Exception as error:
            if debug:
                print(error, "at frame", i)
    
    to_verts = chessboard_3D_verts(board, zofs)
    if "offset" in sensor and sensor["offset"] is not None:
        x_offset, y_offest = sensor["offset"]
        to_verts += np.array([x_offset, y_offest, 0])

    from_verts, non_zero_mask = amalgamate(from_verts_col, median=True)
    to_verts = to_verts[non_zero_mask]

    inlier_mask = get_inlier_mask(from_verts)
    from_verts = from_verts[inlier_mask]
    to_verts = to_verts[inlier_mask]

    if view_from_verts:
        view_verts_3D(from_verts)

    return from_verts, to_verts

def main():
    args = parse_args()
    
    sensors = locations[ args.location ]
    board = boards[ args.board ]

    # Check that we can find the board and it's alignment
    if args.debug:
        for sensor in sensors:
            run_view_board_alignment(sensor, board)
            input("Press enter to continue")

    # load existing data
    clear_all = False
    try:
        calibration_data = json.load( open( args.calibration_file ) )
        for sensor in sensors:
            sid = sensor["serial"]
            calibration_data[sid]['from_verts'] = np.asarray( calibration_data[sid]['from_verts'] )
            calibration_data[sid]['to_verts'] = np.asarray( calibration_data[sid]['to_verts'] )            
        
    except:
        clear_all = True
        
    if clear_all or args.clear:
        calibration_data = {}
        for sensor in sensors:
            sid = sensor["serial"]
            calibration_data[sid] = {}
            calibration_data[sid]['from_verts'] = np.zeros((0,3))
            calibration_data[sid]['to_verts'] = np.zeros((0,3))
    
    # Collect data
    for sensor in sensors:
        sid = sensor["serial"]
        from_verts, to_verts = collect_data(sensor, board, frame_count=args.frame_count, zofs=args.zofs)
        calibration_data[sid]["from_verts"] = np.vstack( (calibration_data[sid]["from_verts"], from_verts) )
        calibration_data[sid]["to_verts"] = np.vstack( (calibration_data[sid]["to_verts"], to_verts) )


    for sensor in sensors:
        sid = sensor["serial"]
        calibration_data[sid]['from_verts'] = calibration_data[sid]['from_verts'].tolist()
        calibration_data[sid]['to_verts'] = calibration_data[sid]['to_verts'].tolist()            
        
    json.dump( calibration_data, open( args.calibration_file,"w"))


if __name__ == "__main__":
    main()
