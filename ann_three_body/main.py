import numpy as np

import physics
import storage
import learning
import visualization
import multiprocessing
from itertools import repeat
import uuid

# The dimensions of storage in this project are as follows:
#                       Example: velocity
# Dimension 0: storage.    [v_x, v_y, vz]
# Dimension 1: body nr  [data_of_mass_0, data_1, data_2, ...]
# Dimension 2: time     [data_t_0, data_t_1, data_t_2, ...]

# Adjust the initial conditions to whatever problem you want to solve

# problem = physics.system.AlphaCentauriSystem()
# problem = physics.system.EearthSunSystem()
# problem = physics.system.EearthSunMoonSystem()
problem = physics.system.RandomNbodySystem(N=3, mass_dev=0., mass_mean=1.)

# Adjust to compare different integrators

# integrator = physics.integrator.OdeIntegrator()
# integrator = physics.integrator.SciPyIvpIntegrator(method='LSODA')
# integrator = physics.integrator.HamiltonianIntegrator()
# integrator = physics.integrator.SymplecticIntegrator()
integrator = physics.integrator.BrutusIntegrator()


def run_and_save_simulation(i, output_folder):
    print(f"\n### Executing simulation {i}")
    problem = physics.system.RandomNbodySystem(N=3, mass_dev=0., mass_mean=1.)

    m = problem.m

    r, v, t = integrator.integrate(10_000, problem)

    energy = physics.energy.system_total(r, v, m)
    relative_error = physics.analytics.get_relative_error(energy).max()
    print(f"Max relative error: {relative_error:.1E}")

    print(f"Saving solution")
    storage.store_simulation(problem.m,
                             problem.r_init,
                             problem.v_init,
                             t,
                             r[-1], v[-1],
                             filename=f"{uuid.uuid1()}",
                             folder=output_folder)


def create_data_set(problem: physics.system.RandomNbodySystem,
                    num=1000, output_folder="output2"):
    # run = lambda i: run_and_save_simulation(problem, integrator, output_folder, i)

    with multiprocessing.Pool() as pool:
        pool.starmap(run_and_save_simulation,
                     zip(range(num), repeat(output_folder)))


def newton_vs_the_machine(problem,
                          integrator,
                          model: learning.BaseModel,
                          num_periods=5,
                          set_timespan=0.05):
    output = model.predict_timesteps(problem.N, problem.m, problem.r_init, problem.v_init,
                                     num_timesteps=num_periods)
    r_ml, v_ml = model.get_data_from_output(output, problem.N)

    integrator.T = num_periods * set_timespan
    r, v, t = integrator.integrate(10_000, problem)

    return r_ml, v_ml, r, v


def resume_fitting(model: learning.BaseModel, datasets, N=3):
    if model is None:
        model = learning.BaseModel.new(N)
    model.fit_model(datasets)


def main():
    integrator.T = 0.05
    # Create the dataset for training the model with

    dataset_folder = f"output/equal_mass_t{integrator.T}_rcm0_pcm0"

    # create_data_set(problem, num=100000000, output_folder=dataset_folder)

    # Train a model
    sets = storage.load_sets_from_folder(dataset_folder)

    layers = 10
    nodes = 128
    model_folder = f"output/2D{layers}Layers{nodes}Nodes{integrator.T}Timestep"

    if not learning.TwoDimensionalModel.exists(model_folder):
        model = learning.TwoDimensionalModel.new(N=3,
                                                 num_layers_hidden=layers, num_nodes=nodes,
                                                 storage_folder=model_folder)
    else:
        model = learning.TwoDimensionalModel.load(path=model_folder)

    resume_fitting(model, sets)

    # Execute Newton vs The Machine: See who performs better
    r_ml, v_ml, r, v = newton_vs_the_machine(problem, integrator,
                                             model,
                                             num_periods=10, set_timespan=integrator.T)

    # Integrate a single problem
    # r, v, t = integrator.integrate(10_000, problem)
    # m = problem.m
    # N = problem.N

    # Visualisation of results
    vis = visualization.show_trajectory(r, v, problem.N, show=False)
    visualization.show_trajectory(r_ml, v_ml, problem.N, *vis,
                                  show=True, alpha=0.45, linestyle='--')
    # visualization.show_energy(r, v, m)
    # visualization.animate_trajectory_2d(r, v, N, m)


if __name__ == "__main__":
    main()
