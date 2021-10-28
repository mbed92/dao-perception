from collections import defaultdict

import numpy as np
from elastica import EndpointForces, PositionVerlet, integrate
from elastica.callback_functions import CallBackBaseClass
from elastica.external_forces import GravityForces
from elastica.rod.cosserat_rod import CosseratRod
from elastica.wrappers import BaseSystemCollection, CallBacks, Constraints, Forcing

GRAVITY = -9.80665
pp_list = defaultdict(list)


class RecordDataCallback(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["position"].append(
                system.position_collection.copy()
            )
            return


def run_model(model, final_time=0.5):
    timestepper = PositionVerlet()
    dt = 1.0e-4
    total_steps = int(final_time / dt)
    integrate(timestepper, model, final_time, total_steps)


def create_model(density=1e3, nu=1e-3, youngs_modulus=1e7, poisson_ratio=0.5):
    class RodSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
        pass

    simulation = RodSimulator()

    shearable_rod = CosseratRod.straight_rod(
        n_elements=50,
        start=np.zeros((3,)),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=0.5,
        base_radius=10 - 2,
        density=density,
        nu=nu,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio
    )

    simulation.append(shearable_rod)

    # Add gravitational forces
    simulation.add_forcing_to(shearable_rod).using(
        GravityForces, acc_gravity=np.array([0.0, GRAVITY, 0.0])
    )

    # Define 1x3 array of the applied forces
    origin_force = np.array([0.0, -GRAVITY * 1.1, -50.0])
    end_force = np.array([0.0, -GRAVITY * 1.1, 50.0])
    simulation.add_forcing_to(shearable_rod).using(
        EndpointForces,  # Traction BC being applied
        origin_force,  # Force vector applied at first node
        end_force,  # Force vector applied at last node
        ramp_up_time=0.01  # Ramp up time
    )

    simulation.collect_diagnostics(shearable_rod).using(
        RecordDataCallback, step_skip=100, callback_params=pp_list
    )

    simulation.finalize()

    return simulation


# NUM_TRAIN_SAMPLES, NUM_VAL_SAMPLES = 500, 50
# x_train, y_train, x_valid, y_valid = list(), list(), list(), list()
# for i in range(NUM_TRAIN_SAMPLES):
#     # create model
#     density = np.random.uniform(900, 1100)
#     nu = np.random.uniform(0.5e-3, 1.5e-3)
#     youngs_modulus = np.random.uniform(1e5, 1e7)
#     poisson_ratio = np.random.uniform(0.4, 0.6)
#     rod = create_model(density, nu, youngs_modulus, poisson_ratio)
#     run_model(rod)
#
#     positions = pp_list["position"]
#     x_train.append(positions[0])
#     y_train.append(positions[-1])
#     pp_list.clear()
#
#     print(f"train {i}")
#
# for i in range(NUM_VAL_SAMPLES):
#     # create model
#     density = np.random.uniform(900, 1100)
#     nu = np.random.uniform(0.5e-3, 1.5e-3)
#     youngs_modulus = np.random.uniform(1e5, 1e7)
#     poisson_ratio = np.random.uniform(0.4, 0.6)
#     rod = create_model(density, nu, youngs_modulus, poisson_ratio)
#     run_model(rod)
#
#     positions = pp_list["position"]
#     x_valid.append(positions[0])
#     y_valid.append(positions[-1])
#     pp_list.clear()
#
#     print(f"valid {i}")
#
# data = {
#     "x_train": np.asarray(x_train),
#     "y_train": np.asarray(y_train),
#     "x_valid": np.asarray(x_valid),
#     "y_valid": np.asarray(y_valid)
# }
#
# np.save("data.npy", data)

from benderopt.base import OptimizationProblem, Observation
from benderopt.optimizer import optimizers

run_model(create_model(970, 1e-3, 5.7e6, 0.5))
s_t = pp_list["position"][0]
s_tplus1 = pp_list["position"][-1]
pp_list.clear()


# We want to minimize the sinus function between 0 and 2pi
def f(density, nu, youngs_modulus, poisson_ratio):
    model = create_model(density, nu, youngs_modulus, poisson_ratio)
    run_model(model)
    distance = np.linalg.norm(s_tplus1 - pp_list["position"][-1], axis=1)
    pp_list.clear()
    return distance.mean()


# We define the parameters we want to optimize:
optimization_problem_data = [
    {
        "name": "density",
        "category": "uniform",
        "search_space": {"low": 900, "high": 1100, "step": 10}
    },
    {
        "name": "nu",
        "category": "uniform",
        "search_space": {"low": 0.5e-3, "high": 1.5e-3, "step": 0.3e-4}
    },
    {
        "name": "youngs_modulus",
        "category": "uniform",
        "search_space": {"low": 1e5, "high": 1e7, "step": 1e3}
    },
    {
        "name": "poisson_ratio",
        "category": "uniform",
        "search_space": {"low": 0.4, "high": 0.6, "step": 0.05}
    }
]

optimization_problem = OptimizationProblem.from_list(optimization_problem_data)
optimizer = optimizers["parzen_estimator"](optimization_problem)
sample = optimizer.suggest()

number_of_evaluation = 5
for _ in range(number_of_evaluation):
    sample = optimizer.suggest()
    loss = f(**sample)
    observation = Observation.from_dict({"loss": loss, "sample": sample})
    optimization_problem.add_observation(observation)

a = optimization_problem.get_best_k_samples(1)
print(f"BEST parameters: {a[0].sample}, loss: {a[0].loss}")
