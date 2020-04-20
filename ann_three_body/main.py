import numpy as np

import physics
import storage
import learning
import visualization

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


def create_data_set(problem: physics.system.MechanicalSystem,
                    integrator: physics.integrator.PhysicsIntegrator,
                    num=1000, output_folder="output2"):
    for i in range(num):
        print(f"\n### Executing simulation {i}")
        problem.set_new()

        r, v, t = integrator.integrate(10_000, problem)
        m = problem.m

        energy = physics.energy.system_total(r, v, m)
        relative_error = physics.analytics.get_relative_error(energy).max()
        print(f"Max relative error: {relative_error:.1E}")

        print(f"Saving solution")
        storage.store_simulation(problem.m, problem.r_init, problem.v_init,
                                 t,
                                 r[-1], v[-1],
                                 filename=f"{i}",
                                 folder=output_folder)


def newton_vs_the_machine(problem, integrator, model, num_periods=10, set_timespan=0.1):
    output = learning.predict_timesteps(model, problem.N, problem.m, problem.r_init, problem.v_init)
    r_ml, v_ml = learning.get_data_from_output(output, problem.N)

    integrator.T = num_periods * set_timespan
    r, v, t = integrator.integrate(10_000, problem)

    return r_ml, v_ml, r, v


def resume_fitting(datasets, model=None, N=3):
    if model is None:
        model = learning.create_model(N)
    learning.fit_model(datasets, model=model, output_dir="output/training_1")


def main():
    # Create the dataset for training the model with
    # create_data_set(problem, integrator, num=100000, output_folder="output/equal_mass_")

    # Train a model
    sets = storage.load_sets_from_folder("output/09-04-2020_equalmass_0.1time")

    # Load a saved model
    model = learning.load_model("output/training_1")
    resume_fitting(sets, model)

    # Execute Newton vs The Machine: See who performs better
    # newton_vs_the_machine(problem, integrator, model)

    # Integrate a single problem
    # r, v, t = integrator.integrate(10_000, problem)
    # m = problem.m
    # N = problem.N

    # Visualisation of results
    # visualization.show_trajectory(r, v, problem.N)
    # visualization.show_trajectory(r_ml, v_ml, problem.N)
    # visualization.show_energy(r, v, m)
    # visualization.animate_trajectory_2d(r, v, N, m)


if __name__ == "__main__":
    main()
