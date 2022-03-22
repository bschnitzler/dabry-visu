import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
import h5py

from font_config import FontsizeConf
from misc import *

state_names = [r"$x\:[m]$", r"$y\:[m]$"]
control_names = [r"$u\:[rad]$"]


class Display:
    """
    Defines all the visualization functions for navigation problems
    """

    def __init__(self, coords='cartesian', mode='only-map'):
        """
        :param coords: Either 'cartesian' for planar problems or 'gcs' for Earth-based problems
        """
        self.coords = coords
        self.mode = mode

        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        self.x_offset = 0.1
        self.y_offset = 0.1

        # Main figure
        self.mainfig = None
        # Single plots figure
        self.spfig = None
        # Adjoint figure
        self.afig = None
        self.map = None
        self.map_adjoint = None
        self.display_setup = False
        self.cm = None
        self.title = 'test'
        self.axes_equal = True

    def setup(self, x_min=-.1, x_max=1.5, y_min=-1., y_max=1.):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        fsc = FontsizeConf()
        plt.rc('font', size=fsc.fontsize)
        plt.rc('axes', titlesize=fsc.axes_titlesize)
        plt.rc('axes', labelsize=fsc.axes_labelsize)
        plt.rc('xtick', labelsize=fsc.xtick_labelsize)
        plt.rc('ytick', labelsize=fsc.ytick_labelsize)
        plt.rc('legend', fontsize=fsc.legend_fontsize)
        plt.rc('font', family=fsc.font_family)
        plt.rc('mathtext', fontset=fsc.mathtext_fontset)

        self.mainfig = plt.figure(num="Navigation problem", constrained_layout=False)
        self.mainfig.subplots_adjust(
            top=0.93,
            bottom=0.08,
            left=0.075,
            right=0.93,
            hspace=0.155,
            wspace=0.13
        )
        self.mainfig.suptitle(self.title)

        if self.mode == "only-map":
            gs = GridSpec(1, 1, figure=self.mainfig)
            self.map = self.mainfig.add_subplot(gs[0, 0])
        elif self.mode == "full":
            """
            In this mode, let the map on the left hand side of the plot and plot the components of the state
            and the control on the right hand side
            """
            print(f'Mode "{self.mode}" unsupported yet')
            exit(1)
            # gs = GridSpec(self.dim_state + self.dim_control, 2, figure=self.mainfig, wspace=.25)
            # self.map = self.mainfig.add_subplot(gs[:, 0])
            # for k in range(self.dim_state):
            #     self.state.append(self.mainfig.add_subplot(gs[k, 1]))
            # for k in range(self.dim_control):
            #     self.control.append(self.mainfig.add_subplot(gs[k + self.dim_state, 1]))
        elif self.mode == "full-adjoint":
            """
            In this mode, plot the state as well as the adjoint state vector
            """
            print(f'Mode "{self.mode}" unsupported yet')
            exit(1)
            # gs = GridSpec(1, 2, figure=self.mainfig, wspace=.25)
            # self.map = self.mainfig.add_subplot(gs[0, 0])
            # self.map_adjoint = self.mainfig.add_subplot(gs[0, 1])  # , projection="polar")
        self.setup_cm()
        self.setup_map()
        if self.mode == "full-adjoint":
            self.setup_map_adj()
        # self.setup_components()
        self.display_setup = True

    def setup_cm(self):
        # top = np.array(colorsys.hls_to_rgb(141 / 360, .6, .81) + (1.,))
        # middle = np.array(colorsys.hls_to_rgb(41 / 360, .6, .88) + (1.,))
        # bottom = np.array(colorsys.hls_to_rgb(358 / 360, .6, .82) + (1.,))
        #
        # S = np.linspace(0., 1., 128)
        #
        # first = np.array([(1 - s) * top + s * middle for s in S])
        # second = np.array([(1 - s) * middle + s * bottom for s in S])
        #
        # newcolors = np.vstack((first, second))
        # newcolors[-1] = np.array([0.4, 0., 1., 1.])
        self.cm = mpl_colors.Colormap('viridis')

    def setup_map(self):
        """
        Sets the display of the map
        """
        CARTESIAN = (self.coords == 'cartesian')
        GCS = (self.coords == 'gcs')
        if CARTESIAN:
            self.map.axhline(y=0, color='k', linewidth=0.5)
            self.map.axvline(x=0, color='k', linewidth=0.5)
            self.map.axvline(x=1., color='k', linewidth=0.5)

        if GCS:
            self.map = Basemap(llcrnrlon=self.x_min - self.x_offset * (self.x_max - self.x_min),
                               llcrnrlat=self.y_min - self.y_offset * (self.y_max - self.y_min),
                               urcrnrlon=self.x_max + self.x_offset * (self.x_max - self.x_min),
                               urcrnrlat=self.y_max + self.y_offset * (self.y_max - self.y_min),
                               rsphere=(6378137.00, 6356752.3142),
                               resolution='l', projection='merc')
            # self.map.shadedrelief()
            self.map.drawcoastlines()
            self.map.fillcontinents()
            # draw parallels
            self.map.drawparallels(np.arange(-80, 80, 17), labels=[1, 0, 0, 0])
            # draw meridians
            self.map.drawmeridians(np.arange(-170, 180, 35), labels=[1, 1, 0, 1])

        if CARTESIAN:
            self.map.set_xlim(self.x_min, self.x_max)
            self.map.set_ylim(self.y_min, self.y_max)
            self.map.set_xlabel('$x$')
            self.map.set_ylabel('$y$')
            self.map.grid(visible=True, linestyle='-.', linewidth=0.5)
            if self.axes_equal:
                self.map.axis('equal')
            self.map.tick_params(direction='in')

    def setup_map_adj(self):
        """
        Sets the display of the map for the adjoint state
        """
        self.map_adjoint.set_xlim(-1.1, 1.1)
        self.map_adjoint.set_ylim(-1.1, 1.1)

        self.map_adjoint.set_xlabel('$p_x\;[s/m]$')
        self.map_adjoint.set_ylabel('$p_y\;[s/m]$')

        self.map_adjoint.grid(visible=True, linestyle='-.', linewidth=0.5)
        self.map_adjoint.tick_params(direction='in')
        if self.axes_equal:
            self.map_adjoint.axis('equal')

    def set_title(self, title):
        self.title = title
        self.mainfig.suptitle(self.title)

    def set_wind_density(self, level: int):
        """
        Sets the wind vector density in the plane.
        :param level: The density level
        """
        self.nx_wind *= level
        self.ny_wind *= level

    def draw_wind(self, filepath):
        with h5py.File(filepath, 'r') as f:
            nt, nx, ny, _ = f['data'].shape
            X = np.zeros((nx, ny))
            Y = np.zeros((nx, ny))
            X[:] = f['grid'][:, :, 0]
            Y[:] = f['grid'][:, :, 1]
            self.x_min = np.min(X)
            self.x_max = np.max(X)
            self.y_min = np.min(Y)
            self.y_max = np.max(Y)
            self.setup_map()

            # Resize window

            # X, Y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.nx_wind),
            #                    np.linspace(self.y_min, self.y_max, self.ny_wind))
            #
            # cartesian = np.dstack((X, Y)).reshape(-1, 2)
            #
            # uv = np.array(list(map(self.windfield, list(cartesian))))
            U = f['data'][0, :, :, 0].flatten()
            V = f['data'][0, :, :, 1].flatten()

            norms = np.sqrt(U ** 2 + V ** 2)
            lognorms = np.log(np.sqrt(U ** 2 + V ** 2))

            norm = mpl_colors.Normalize()
            norm.autoscale(np.array([0., 2.]))

            sm = mpl_cm.ScalarMappable(cmap=self.cm, norm=norm)

            # sm.set_array(np.array([]))

            def color_value(x):
                res = sm.to_rgba(x)
                if x > 1.:
                    return res
                else:
                    newres = res[0], res[1], res[2], 0.3
                    return newres

            # color = list(map(lambda x: color_value(x), norms))
            kwargs = {'headwidth':2, 'width':0.004}
            if self.coords == 'gcs':
                kwargs['latlon'] = True
            self.map.quiver(X, Y, U / norms, V / norms, **kwargs)# color=color)

            # divider = make_axes_locatable(self.map)
            # cax = divider.append_axes("right", size="5%", pad=0.05)

            # cb = plt.colorbar(sm, ax=[self.map], location='right')
            # cb = plt.colorbar(sm, cax=cax)
            # cb.set_label('$|v_w|\;[m/s]$')
            # cb.ax.semilogy()
            # cb.ax.yaxis.set_major_formatter(mpl_ticker.LogFormatter())#mpl_ticker.FuncFormatter(lambda s, pos: (np.exp(s*np.log(10)), pos)))

    def setup_components(self):
        for k, state_plot in enumerate(self.state):
            state_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
            state_plot.tick_params(direction='in')
            state_plot.yaxis.set_label_position("right")
            state_plot.yaxis.tick_right()
            state_plot.set_ylabel(state_names[k])
            plt.setp(state_plot.get_xticklabels(), visible=False)

        for k, control_plot in enumerate(self.control):
            control_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
            control_plot.tick_params(direction='in')
            control_plot.yaxis.set_label_position("right")
            control_plot.yaxis.tick_right()
            control_plot.set_ylabel(control_names[k])
            # Last plot
            if k == len(self.control) - 1:
                control_plot.set_xlabel(r"$t\:[s]$")

    def plot_traj(self, points, controls, ts, type, last_index, interrupted, color_mode="default", **kwargs):
        """
        Plots the given trajectory according to selected display mode
        """
        t_tick = (ts[-1] - ts[0]) / 5.
        if not self.display_setup:
            self.setup()
        label = None
        s = 0.5 * np.ones(last_index)
        if color_mode == "default":
            colors = None
            cmap = None
        elif color_mode == "monocolor":
            colors = np.tile(monocolor_colors[type], last_index).reshape((last_index, 4))
            cmap = None
        elif color_mode == "reachability":
            colors = np.ones((last_index, 4))
            colors[:] = np.einsum("ij,j->ij", colors, reachability_colors[type]["steps"])

            t_count = 0.
            for k, t in enumerate(ts):
                if t - t_count > t_tick:
                    t_count = t
                    try:
                        colors[k] = reachability_colors[type]["time-tick"]
                        s[k] = 1.5
                    except IndexError:
                        pass
            colors[-1, :] = reachability_colors[type]["last"]  # Last point
            s[-1] = 2.
            cmap = plt.get_cmap("YlGn")
        elif color_mode == "reachability-enhanced":
            if "scalar_prods" not in kwargs:
                raise ValueError('Expected "scalar_prods" argument for "reachability-enhanced" plot mode')
            _cmap = plt.get_cmap('winter')
            colors = np.ones((last_index, 4))
            t_count = 0.
            for k, t in enumerate(ts):
                if t - t_count > 0.5:
                    t_count = t
                    colors[k] = reachability_colors[type]["time-tick"]
                    s[k] = 2.
            for k in range(last_index):
                colors[k:] = _cmap(kwargs['scalar_prods'][k])
            colors[-1, :] = reachability_colors[type]["last"]  # Last point
            s *= 1.5
            s[-1] = 2.
            cmap = plt.get_cmap("YlGn")
        else:
            raise ValueError(f"Unknown plot mode {color_mode}")
        s *= 3.

        self.map.scatter(points[:last_index - 1, 0], points[:last_index - 1, 1],
                         s=s[:-1],
                         c=colors[:-1],
                         cmap=cmap,
                         label=label,
                         marker=None, latlon=True)
        self.map.scatter(points[last_index - 1, 0], points[last_index - 1, 1],
                         s=10. if interrupted else s[-1],
                         c=[colors[-1]],
                         marker=(r'x' if interrupted else 'o'),
                         linewidths=1., latlon=True)
        # if controls:
        #     dt = np.mean(ts[1:] - ts[:-1])
        #     for k, point in enumerate(points):
        #         u = controls[k]
        #         _s = np.array([np.cos(u), np.sin(u)]) * dt
        #         self.map.arrow(point[0], point[1], _s[0], _s[1], width=0.0001, color=colors[k])
        # if self.mode == "full":
        #     for k in range(points.shape[1]):
        #         self.state[k].plot(ts[:last_index], points[:last_index, k])
        #     k = 0
        #     self.control[k].plot(ts[:last_index], traj.controls[:last_index])
        # elif self.mode == "full-adjoint":
        #     if isinstance(traj, AugmentedTraj):
        #         self.map_adjoint.scatter(traj.adjoints[:last_index, 0], traj.adjoints[:last_index, 1],
        #                                  s=s,
        #                                  c=colors,
        #                                  cmap=cmap,
        #                                  label=label,
        #                                  marker=None)

    def draw_trajs(self, filepath):
        with h5py.File(filepath, 'r') as f:
            for traj in f.values():
                self.plot_traj(traj['data'],
                               traj['controls'],
                               traj['ts'],
                               traj.attrs['type'],
                               traj.attrs['last_index'],
                               traj.attrs['interrupted'], color_mode='reachability')

    def draw_rff(self, filepath):
        with h5py.File(filepath, 'r') as f:
            self.map.contourf(f['grid'][:, :, 0], f['grid'][:, :, 1], f['data'][5, :, :], [-1e-3, 1e-3])

if __name__ == '__main__':
    """
    d = Display(coords='gcs')
    d.setup(x_min=-76., x_max=5., y_min=35., y_max=52.)
    # d.draw_wind('/home/bastien/Documents/data/wind/mermoz/Vancouver-Honolulu-1.0/data.h5')
    d.draw_trajs('/home/bastien/Documents/work/mermoz/output/trajs/trajectories.h5')
    d.map.drawgreatcircle(-75., 40., 2., 48., linewidth=1, color='b', alpha=0.4,
                           linestyle='--',
                           label='Great circle')
    """
    d = Display(coords='cartesian')
    d.setup()
    #d.draw_rff('/home/bastien/Documents/work/mermoz/output/rff/rff.h5')
    d.draw_wind('/home/bastien/Documents/data/wind/mermoz/Dakar-Natal-0.5-tweaked/data.h5')
    d.draw_rff('/home/bastien/Documents/data/rff/test.h5')
    plt.show()
