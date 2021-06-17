import os

from matplotlib import pylab, use

use('TKAgg')

import world

NUM_EPISODES = 10
EPISODE_LENGTH = 100
SHOW_INTERVAL = 10
ENV_CONFIG = {
    "plane_position": [0, 0, -0.3],
    "plane_quaternion": [0, 0, 0, 1],
    "pusher_position": [0, -0.5, 0],
    "pusher_quaternion": [0, 0, 0, 1],
    "pusher_global_scaling": 0.5,
    "cam_target_pos": [0, 0.1, 0.5],
    "cam_roll": 0,
    "cam_pitch": -90,
    "cam_yaw": 0,
    "cam_distance": 1.5,
    "projection_w": 640,
    "projection_h": 480,
    "near_plane": 0.01,
    "far_plane": 100,
    "fov": 60,
    "additional_objects_folder": None
    # "additional_objects_folder": os.path.join(os.path.dirname(__file__), '..', '..', 'objects')
}

if __name__ == "__main__":
    myenv = world.environment.pusher.PusherEnv(ENV_CONFIG)
    fig = pylab.figure()

    for _ in range(NUM_EPISODES):
        myenv.reset()

        for i in range(EPISODE_LENGTH):
            myenv.step()

            if i % SHOW_INTERVAL == 0:
                img = myenv.get_camera_image()
                pylab.imshow(img, interpolation='none', animated=True, label="pybullet")
                pylab.show(block=False)
                pylab.pause(0.01)

    myenv.stop_sim()
