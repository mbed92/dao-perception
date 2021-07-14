import pybullet as p
import pybullet_data as pd

p.connect(p.GUI)
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setAdditionalSearchPath("/home/mbed/Projects/rl-physnet/config")
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)

model = p.loadURDF("pisa_foot/pisa_foot.urdf", [0, 0, 1], [0, 0, 0, 1])

while (1):
    pass

# restitutionId = p.addUserDebugParameter("restitution", 0, 1, 0.5)
#
# localInertiaDiagonalXId = p.addUserDebugParameter("localInertiaDiagonalX", 0, 3, 0.2)
# localInertiaDiagonalYId = p.addUserDebugParameter("localInertiaDiagonalY", 0, 3, 0.2)
# localInertiaDiagonalZId = p.addUserDebugParameter("localInertiaDiagonalZ", 0, 3, 0.2)
#
# lateralFrictionId = p.addUserDebugParameter("lateral friction", 0, 1, 0.5)
# spinningFrictionId = p.addUserDebugParameter("spinning friction", 0, 1, 0.03)
# rollingFrictionId = p.addUserDebugParameter("rolling friction", 0, 1, 0.03)
#
# plane = p.loadURDF("plane.urdf", [0, 0, 0])
# aaa = p.loadURDF("cube.urdf", [0, 0, 3], flags=p.URDF_USE_SELF_COLLISION)
# p.changeDynamics(aaa, -1, mass=0.1)
#
#
# def _load_softbody(basePos):
#     return p.loadSoftBody("cube.obj", basePosition=basePos, scale=0.5, mass=0.1, useNeoHookean=0,
#                           useBendingSprings=1, useMassSpring=1, springElasticStiffness=40, springDampingStiffness=40,
#                           springDampingAllDirections=1, useSelfCollision=0, frictionCoeff=10.0, useFaceContact=1,
#                           collisionMargin=0.001)
#     # return p.loadSoftBody("sphere_smooth.obj", basePosition=basePos, scale=0.5, mass=0.1, useNeoHookean=1,
#     #                       NeoHookeanMu=400, NeoHookeanLambda=600, NeoHookeanDamping=0.001, useSelfCollision=1,
#     #                       frictionCoeff=.5, collisionMargin=0.001)
#
# sphere = _load_softbody([0, 0, 1])
#
# p.changeDynamics(sphere, -1, mass=0.1)
# p.setRealTimeSimulation(1)
# p.setGravity(0, 0, -10)
#
# while (1):
#     restitution = p.readUserDebugParameter(restitutionId)
#     lidX = p.readUserDebugParameter(localInertiaDiagonalXId)
#     lidY = p.readUserDebugParameter(localInertiaDiagonalYId)
#     lidZ = p.readUserDebugParameter(localInertiaDiagonalZId)
#
#     localInertiaDiagonal = [lidX, lidY, lidZ]
#     lateralFriction = p.readUserDebugParameter(lateralFrictionId)
#     spinningFriction = p.readUserDebugParameter(spinningFrictionId)
#     rollingFriction = p.readUserDebugParameter(rollingFrictionId)
#     p.changeDynamics(plane, -1, lateralFriction=lateralFriction)
#     p.changeDynamics(plane, -1, restitution=restitution)
#     p.changeDynamics(sphere, -1, lateralFriction=lateralFriction)
#     p.changeDynamics(sphere, -1, spinningFriction=spinningFriction)
#     p.changeDynamics(sphere, -1, rollingFriction=rollingFriction)
#     p.changeDynamics(sphere, -1, restitution=restitution)
#     p.changeDynamics(sphere, -1, localInertiaDiagonal=localInertiaDiagonal)
