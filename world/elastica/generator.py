from collections import defaultdict

import numpy as np

import utils
from elastica import EndpointForces, PositionVerlet, integrate
from elastica.callback_functions import CallBackBaseClass
from elastica.external_forces import GravityForces
from elastica.rod.cosserat_rod import CosseratRod
from elastica.wrappers import BaseSystemCollection, CallBacks, Constraints, Forcing

GRAVITY = -9.80665
PP_LIST = defaultdict(list)


class RodSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


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


def create_model(n_elements=50,
                 start=np.zeros((3,)), direction=np.array([0.0, 0.0, 1.0]), normal=np.array([0.0, 1.0, 0.0]),
                 base_length=0.5, base_radius=0.025, density=1e3, nu=1e-3, youngs_modulus=1e7, poisson_ratio=0.5,
                 step_skip=100,
                 origin_force=np.array([0.0, -GRAVITY, 0.0]), end_force=np.array([0.0, -GRAVITY, 0.0])):
    simulation = RodSimulator()

    shearable_rod = CosseratRod.straight_rod(
        n_elements=n_elements,
        start=start,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
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
    simulation.add_forcing_to(shearable_rod).using(
        EndpointForces,  # Traction BC being applied
        origin_force,  # Force vector applied at first node
        end_force,  # Force vector applied at last node
        ramp_up_time=0.01  # Ramp up time
    )

    simulation.collect_diagnostics(shearable_rod).using(
        RecordDataCallback, step_skip=step_skip, callback_params=PP_LIST
    )

    simulation.finalize()
    return simulation


def run_model(model, final_time=0.5, dt=1.0e-4):
    timestepper = PositionVerlet()
    total_steps = int(final_time / dt)
    integrate(timestepper, model, final_time, total_steps)


def randomized_dataset(config, save_file="dataset.npy"):
    dataset = defaultdict()

    for dataset_name in config["dataset_keys"]:
        sequences = list()
        sequences_params = list()

        for step in range(config[dataset_name]["num_sequences"]):
            n_elements = config["common"]["rod_num_elements"]
            start = np.asarray(config["common"]["rod_start_position"])
            direction = np.asarray(config["common"]["rod_direction"])
            normal = np.asarray(config["common"]["rod_normal"])
            base_length = config["common"]["rod_base_length"]
            base_radius = config["common"]["rod_base_radius"]
            density = np.random.uniform(*config[dataset_name]["rod_density_range"])
            nu = np.random.uniform(*config[dataset_name]["rod_energy_dissipation_range"])
            youngs_modulus = np.random.uniform(*config[dataset_name]["rod_youngus_modulus_range"])
            poisson_ratio = np.random.uniform(*config[dataset_name]["rod_poisson_ratio"])
            step_skip = config["common"]["sequence_dstep"]
            origin_force = np.random.uniform(*config[dataset_name]["rod_origin_force_range"], 3)
            end_force = np.random.uniform(*config[dataset_name]["rod_end_force_range"], 3)

            # create a simulation
            model = create_model(n_elements=n_elements,
                                 start=start,
                                 direction=direction,
                                 normal=normal,
                                 base_length=base_length,
                                 base_radius=base_radius,
                                 density=density,
                                 nu=nu,
                                 youngs_modulus=youngs_modulus,
                                 poisson_ratio=poisson_ratio,
                                 step_skip=step_skip,
                                 origin_force=origin_force,
                                 end_force=end_force)

            # run model and get positions of a rod
            run_model(model,
                      config["common"]["sequence_length"],
                      config["common"]["sequence_dt"])

            sequences.append(np.asarray(PP_LIST["position"]))
            sequences_params.append({
                "n_elements": n_elements, "base_length": base_length, "base_radius": base_radius, "density": density,
                "nu": nu, "youngs_modulus": youngs_modulus, "poisson_ratio": poisson_ratio,
                "origin_force": origin_force, "end_force": end_force
            })

            # create gifs from random samples (WARNING: if dt is large gifs can be large and creating it will be slow!)
            if config["common"]["save_random_samples_gifs"] and step % 100 == 0:
                utils.visualize.visualize_rod_3d(PP_LIST["position"], output_filename=f"file_{dataset_name}_{step}.gif")

            PP_LIST.clear()

        dataset[dataset_name] = {
            "sequences": sequences,
            "sequences_params": sequences_params
        }

    # save a dataset
    np.save(save_file, dataset)
