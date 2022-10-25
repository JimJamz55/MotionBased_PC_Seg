import pyrealsense2 as rs

pipe = rs.pipeline()
pipe.start()

while True:
    frames = pipe.wait_for_frames()
    depth = frames.get_depth_frame()
    width = depth.get_width()
    height = depth.get_height()
    dist = depth.get_distance(int(width/2), int(height/2))
    print("Facing an object {0:.3} meters away".format(dist))

pipe.stop()
