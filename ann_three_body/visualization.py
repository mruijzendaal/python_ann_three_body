import matplotlib.widgets
import mpl_toolkits.axes_grid1
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import physics.energy


def show_energy(r, v, m):
    E_system = physics.energy.system_total(r, v, m)
    E_gravitational = physics.energy.gravitational(r, m).sum(axis=1)
    E_kinetic = physics.energy.kinetic(v, m).sum(axis=1)

    for energy, label in [(E_gravitational, "Gravitational energy"),
                          (E_kinetic, "Kinetic energy"),
                          (E_system, "Total energy"),
                          ]:
        plt.plot(energy)
        plt.title(f"{label}")
        plt.xlabel("Energy [N]")
        plt.ylabel("Time [-]")
        plt.show()

        plt.plot((energy - energy[0]) / energy[0])
        plt.title(f"Error in {label}")
        plt.xlabel("Relative energy error [-]")
        plt.ylabel("Time [-]")
        plt.show()


def show_trajectory(r, v, N):
    r2d = r[:, :, :2]
    v2d = v[:, :, :2]

    # Create figure
    fig = plt.figure(figsize=(15, 15))

    #
    #   Plot 3D
    #
    # Create 3D axes
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_zlabel("z-coordinate", fontsize=14)
    # Plot the orbits
    # ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="darkblue")
    # ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="tab:red")
    # # Plot the final positions of the stars
    # ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="darkblue", marker="o", s=100, label="Alpha Centauri A")
    # ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="tab:red", marker="o", s=100, label="Alpha Centauri B")

    #
    #   Plot 2D
    #
    # Plot the orbits
    # Create 3D axes
    ax = fig.add_subplot(111)

    for i in range(N):
        c = ["darkblue", "tab:red", "green"][i]
        # Plot the orbits
        x = r2d[:, i, 0].flatten()
        y = r2d[:, i, 1].flatten()

        ax.plot(x, y, color=c)
        # Plot the final positions of the stars
        ax.scatter(x[-1], y[-1], color=c, marker="o", s=100, label="Alpha Centauri A")

    # Add a few more bells and whistles
    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylabel("y-coordinate", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
    ax.legend(loc="upper left", fontsize=14)

    fig.show()


def animate_trajectory_2d(r, v, N, m):
    r2d = r[:, :, :2]
    v2d = v[:, :, :2]

    r = r2d

    matplotlib.use("TkAgg")
    fig, ax = plt.subplots()

    lines = []
    scatters = []
    for i in range(N):
        line, = ax.plot([], [], lw=1, label=f"Mass {i} = {m[i, 0]:.3f}")
        lines.append(line)
        scat = ax.scatter([0], [0], color=line.get_color())
        scatters.append(scat)

    ax.set_xlim(r[:, :, 0].min(), r[:, :, 0].max())
    ax.set_ylim(r[:, :, 1].min(), r[:, :, 1].max())

    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylabel("y-coordinate", fontsize=14)
    ax.set_title(f"Visualization of orbits of stars in a {N}-body system\n", fontsize=14)
    ax.legend(loc="upper left", fontsize=14)

    def init():
        for line in lines:
            line.set_data([], [])
        # for scat in scatters:
        # scat.set_offsets([[], []])
        return lines + scatters

    N_frames = r.shape[0]
    fps = 50
    T = 10
    multiplier = N_frames / (fps * T)

    # animation function.  This is called sequentially
    def animate(i):
        for p, line in enumerate(lines):
            x, y = r[:1 + int(i * multiplier), p, :].T
            line.set_data(x, y)
            scatters[p].set_offsets(np.c_[x[-1], y[-1]])
        return lines + scatters

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=fps * T, interval=1e3 // fps, blit=False,
                         repeat=True)
    plt.show()
