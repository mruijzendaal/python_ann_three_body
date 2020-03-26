import matplotlib.widgets
import mpl_toolkits.axes_grid1
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import physics.energy


class Player(FuncAnimation):
    """
    This class makes it easy to create an interactive Matplotlib figure
    with a time-revolution.
    Example usage:

    ### using this class is as easy as using FuncAnimation:
    # fig, ax = plt.subplots()
    # x = np.linspace(0,6*np.pi, num=100)
    # y = np.sin(x)
    #
    # ax.plot(x,y)
    # point, = ax.plot([],[], marker="o", color="crimson", ms=15)
    #
    # def update(i):
    #     point.set_data(x[i],y[i])
    #
    # ani = Player(fig, update, maxi=len(y)-1)
    #
    # plt.show()

    Credits to https://stackoverflow.com/a/46327978/1644301
    """

    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.update, frames=self.play(),
                               init_func=init_func, fargs=fargs,
                               save_count=save_count, **kwargs)

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.slider = matplotlib.widgets.Slider(sliderax, '',
                                                self.min, self.max, valinit=self.i)
        self.button_oneback = matplotlib.widgets.Button(playerax, label=r'$\u29CF$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label=r'$\u29D0$')
        self.button_forward = matplotlib.widgets.Button(fax, label=r'$\u25B6$')
        self.button_stop = matplotlib.widgets.Button(sax, label=r'$\u25A0$')
        self.button_back = matplotlib.widgets.Button(bax, label=r'$\u25C0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self, i):
        self.slider.set_val(i)


def show_energy(r, v, m):
    E_system = physics.energy.system_total(r, v, m)
    plt.plot(E_system / E_system[0])
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


def animate_trajectory_2d(r, v, N):
    r2d = r[:, :, :2]
    v2d = v[:, :, :2]

    r = r2d

    matplotlib.use("TkAgg")
    fig, ax = plt.subplots()

    lines = []
    scatters = []
    for i in range(N):
        line, = ax.plot([], [], lw=1)
        lines.append(line)
        scat = ax.scatter([0], [0], color=line.get_color())
        scatters.append(scat)

    ax.set_xlim(r[:, :, 0].min(), r[:, :, 0].max())
    ax.set_ylim(r[:, :, 1].min(), r[:, :, 1].max())

    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylabel("y-coordinate", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
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
