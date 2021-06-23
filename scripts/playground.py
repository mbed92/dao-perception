# a mimic joint can act as a gear between two joints
# you can control the gear ratio in magnitude and sign (>0 reverses direction)

import time

import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setGravity(0, 0, -10.0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# create pusher
p.createCollisionShape(p.GEOM_PLANE)
plane = p.createMultiBody(0, 0)
base = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
link = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.05, 0.05])
pusher = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.1, 0.2])

baseMass = 10.0
baseCollisionShapeIndex = base
baseVisualShapeIndex = -1
basePosition = [0, 0, 1]
baseOrientation = [0, 0, 0, 1]
linkMasses = [0.5, 0.5]
linkCollisionShapeIndices = [link, pusher]
linkVisualShapeIndices = [link, pusher]
linkPositions = [[0.3, 0, -0.25], [0.15, 0, 0]]
linkOrientations = [[0, 0, 0, 1]] * len(linkMasses)
linkInertialFramePositions = [[0, 0, 0]] * len(linkMasses)
linkInertialFrameOrientations = [[0, 0, 0, 1]] * len(linkMasses)
linkParentIndices = [0, 1]
linkJointTypes = [p.JOINT_REVOLUTE, p.JOINT_SPHERICAL]
linkJointAxis = [[0, 0, 1], [1, 1, 0]]

pusher_id = p.createMultiBody(baseMass=baseMass,
                              baseCollisionShapeIndex=baseCollisionShapeIndex,
                              baseVisualShapeIndex=baseVisualShapeIndex,
                              basePosition=basePosition,
                              baseOrientation=baseOrientation,
                              linkMasses=linkMasses,
                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                              linkVisualShapeIndices=linkVisualShapeIndices,
                              linkPositions=linkPositions,
                              linkOrientations=linkOrientations,
                              linkInertialFramePositions=linkInertialFramePositions,
                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                              linkParentIndices=linkParentIndices,
                              linkJointTypes=linkJointTypes,
                              linkJointAxis=linkJointAxis)

c = p.createConstraint(parentBodyUniqueId=plane,
                       parentLinkIndex=-1,
                       childBodyUniqueId=pusher_id,
                       childLinkIndex=-1,
                       jointType=p.JOINT_FIXED,
                       jointAxis=[0, 0, 1],
                       parentFramePosition=[0, 0, 1],
                       childFramePosition=[0, 0, 0],
                       parentFrameOrientation=[0, 0, 0, 1],
                       childFrameOrientation=[0, 0, 0, 1]
                       )
p.setJointMotorControl2(pusher_id, 0, p.POSITION_CONTROL, targetPosition=0.2, force=10, maxVelocity=3)
p.setJointMotorControl2(pusher_id, 1, p.POSITION_CONTROL, targetPosition=1.0, force=10, maxVelocity=3)

while (1):
    time.sleep(1. / 240.)
    p.stepSimulation()
