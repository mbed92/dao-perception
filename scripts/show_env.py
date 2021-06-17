import time

from matplotlib import pylab

import world

if __name__ == "__main__":
    myenv = world.environment.pusher.PusherEnv()
    myenv.randomize_environment()

    for i in range(5):
        img = myenv.get_camera_image()
        pylab.imshow(img, interpolation='none', animated=True, label="pybullet")
        pylab.show()
        time.sleep(1)

    myenv.stop_sim()
