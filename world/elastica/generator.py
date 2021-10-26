from collections import defaultdict

import numpy as np
import utils
from elastica import EndpointForces, PositionVerlet, integrate
from elastica.callback_functions import CallBackBaseClass
from elastica.external_forces import GravityForces
from elastica.rod.cosserat_rod import CosseratRod
from elastica.wrappers import BaseSystemCollection, CallBacks, Constraints, Forcing

GRAVITY = -9.80665


class RodSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


class RecordDataCallback(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(
                system.position_collection.copy()
            )
            return


def create_simulation(config, dataset_name, step_skip):
    simulation = RodSimulator()
    density = np.random.uniform(*config[dataset_name]["rod_density_range"])
    nu = np.random.uniform(*config[dataset_name]["rod_energy_dissipation_range"])
    E = np.random.uniform(*config[dataset_name]["rod_youngus_modulus_range"])
    poisson_ratio = np.random.uniform(*config[dataset_name]["rod_poisson_ratio"])

    shearable_rod = CosseratRod.straight_rod(
        n_elements=config["common"]["rod_num_elements"],
        start=np.asarray(config["common"]["rod_start_position"]),
        direction=np.asarray(config["common"]["rod_direction"]),
        normal=np.asarray(config["common"]["rod_normal"]),
        base_length=config["common"]["rod_base_length"],
        base_radius=config["common"]["rod_base_radius"],
        density=density,
        nu=nu,
        youngs_modulus=E,
        poisson_ratio=poisson_ratio,
    )

    simulation.append(shearable_rod)

    # Add gravitational forces
    simulation.add_forcing_to(shearable_rod).using(
        GravityForces, acc_gravity=np.array([0.0, GRAVITY, 0.0])
    )

    # Define 1x3 array of the applied forces
    origin_force = np.array([0.0, -GRAVITY * 1.1, -10.0])
    end_force = np.array([0.0, -GRAVITY * 1.1, 10.0])
    simulation.add_forcing_to(shearable_rod).using(
        EndpointForces,  # Traction BC being applied
        origin_force,  # Force vector applied at first node
        end_force,  # Force vector applied at last node
        ramp_up_time=0.01  # Ramp up time
    )

    pp_list = defaultdict(list)
    simulation.collect_diagnostics(shearable_rod).using(
        RecordDataCallback, step_skip=step_skip, callback_params=pp_list
    )

    simulation.finalize()
    return simulation, pp_list


def randomized_dataset(config):
    final_time = config["common"]["sequence_length"]
    dt = config["common"]["sequence_dt"]
    dstep = config["common"]["sequence_dstep"]
    total_steps = int(final_time / dt)
    step_skip = int(total_steps / dstep)
    print(f"Each sequence will be {total_steps} steps long.")

    for dataset_name in config["dataset_keys"]:
        for step in range(config[dataset_name]["num_sequences"]):
            sim, pp = create_simulation(config, dataset_name, step_skip)
            integrate(PositionVerlet(), sim, final_time, total_steps)
            utils.visualize.visualize_rod_3d(pp["position"])
            assert  2 == 1

            # import pickle
            # filename = "continuum_snake.dat"
            # file = open(filename, "wb")
            # pickle.dump(pp_list, file)
            # file.close()
