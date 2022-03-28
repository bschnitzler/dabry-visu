import colorsys
import json
import os
import shutil
import webbrowser

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from IPython.lib.display import IFrame
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
import h5py
import scipy.ndimage
from math import floor
from geopy import Nominatim
import datetime
import markdown
from IPython.core.display import display, HTML

from mdisplay.font_config import FontsizeConf
from mdisplay.geodata import GeoData
from mdisplay.misc import *
from mdisplay.params_summary import ParamsSummary

state_names = [r"$x\:[m]$", r"$y\:[m]$"]
control_names = [r"$u\:[rad]$"]


class Display:
    """
    Defines all the visualization functions for navigation problems
    """

    def __init__(self, coords='cartesian', mode='only-map', title='Title', projection='merc'):
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
        self.title = title
        self.axes_equal = True
        self.projection = projection

        self.cm_norm_min = None
        self.cm_norm_max = None

        self.output_path = None
        self.params_fname = 'params.json'
        self.params_ss_path = '/home/bastien/Documents/work/mdisplay/data'
        self.params_ss_fname = 'params.css'

        self.geodata = GeoData()

    def setup(self, bl=None, tr=None, bl_off=None, tr_off=None):

        if bl is None:
            with open(os.path.join(self.output_path, self.params_fname), 'r') as f:
                params = json.load(f)
            x_min = params['bl_wind'][0]
            x_max = params['tr_wind'][0]
            y_min = params['bl_wind'][1]
            y_max = params['tr_wind'][1]
        else:
            if type(bl) == str:
                x_min, y_min = self.geodata.get_coords(bl)
            else:
                x_min = bl[0]
                y_min = bl[1]

            if type(tr) == str:
                x_max, y_max = self.geodata.get_coords(tr)
            else:
                x_max = tr[0]
                y_max = tr[1]

        if bl_off is not None:
            self.x_offset = bl_off

        if tr_off is not None:
            self.y_offset = tr_off

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
        # Windy default cm
        cm_values = [[0, [98, 113, 183, 255]],
                     [1, [57, 97, 159, 255]],
                     [3, [74, 148, 169, 255]],
                     [5, [77, 141, 123, 255]],
                     [7, [83, 165, 83, 255]],
                     [9, [53, 159, 53, 255]],
                     [11, [167, 157, 81, 255]],
                     [13, [159, 127, 58, 255]],
                     [15, [161, 108, 92, 255]],
                     [17, [129, 58, 78, 255]],
                     [19, [175, 80, 136, 255]],
                     [21, [117, 74, 147, 255]],
                     [24, [109, 97, 163, 255]],
                     [27, [68, 105, 141, 255]],
                     [29, [92, 144, 152, 255]],
                     [36, [125, 68, 165, 255]],
                     [46, [231, 215, 215, 256]],
                     [51, [219, 212, 135, 256]],
                     [77, [205, 202, 112, 256]],
                     [104, [128, 128, 128, 255]]]
        # Truncated Windy cm
        cm_values = [[0, [98, 113, 183, 255]],
                     [1, [57, 97, 159, 255]],
                     [3, [74, 148, 169, 255]],
                     [5, [77, 141, 123, 255]],
                     [7, [83, 165, 83, 255]],
                     [9, [53, 159, 53, 255]],
                     [11, [167, 157, 81, 255]],
                     [13, [159, 127, 58, 255]],
                     [15, [161, 108, 92, 255]],
                     [17, [129, 58, 78, 255]],
                     [19, [175, 80, 136, 255]],
                     [21, [117, 74, 147, 255]],
                     [24, [109, 97, 163, 255]],
                     [27, [68, 105, 141, 255]],
                     [29, [92, 144, 152, 255]],
                     [36, [125, 68, 165, 255]]]
        self.cm_norm_min = 0.
        self.cm_norm_max = 36.

        def lighten(c):
            hls = colorsys.rgb_to_hls(*(np.array(c[:3]) / 256.))
            hls = (hls[0], 0.5 + 0.5 * hls[1], 0.6 + 0.4 * hls[2])
            res = list(colorsys.hls_to_rgb(*hls)) + [c[3] / 256.]
            return res

        newcolors = np.array(lighten(cm_values[0][1]))
        for ii in range(len(cm_values) - 1):
            j_min = 10 * cm_values[ii - 1][0]
            j_max = 10 * cm_values[ii][0]
            for j in range(j_min, j_max):
                c1 = np.array(lighten(cm_values[ii - 1][1]))
                c2 = np.array(lighten(cm_values[ii][1]))
                t = (j - j_min) / (j_max - j_min)
                newcolors = np.vstack((newcolors, (1 - t) * c1 + t * c2))
        self.cm = mpl_colors.ListedColormap(newcolors, name='Windy')

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
                               # rsphere=(6378137.00, 6356752.3142),
                               resolution='l', projection=self.projection)
            # self.map.shadedrelief()
            self.map.drawcoastlines()
            self.map.fillcontinents()
            # draw parallels
            lat_min = min(self.y_min, self.y_max)
            lat_max = max(self.y_min, self.y_max)
            n_lat = floor((lat_max - lat_min) / 10) + 2
            self.map.drawparallels(10. * (floor(lat_min / 10.) + np.arange(n_lat)), labels=[1, 0, 0, 0])
            # draw meridians
            lon_min = min(self.x_min, self.x_max)
            lon_max = max(self.x_min, self.x_max)
            n_lon = floor((lon_max - lon_min) / 10) + 2
            self.map.drawmeridians(10. * (floor(lon_min / 10.) + np.arange(n_lon)), labels=[1, 1, 0, 1])

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

    def draw_point_by_name(self, name):
        loc = self.geodata.get_coords(name)
        self.map.scatter(loc[0], loc[1], s=8., color='red', marker='D', latlon=True, zorder=ZO_ANNOT)
        plt.annotate(name, self.map(loc[0], loc[1]), (10, 10), textcoords='offset pixels', ha='center')

    def draw_wind(self, filename, adjust_map=False):

        # Wind is piecewise constant
        wind_nointerp = True

        with h5py.File(os.path.join(self.output_path, filename), 'r') as f:
            nt, nx, ny, _ = f['data'].shape
            X = np.zeros((nx, ny))
            Y = np.zeros((nx, ny))
            X[:] = f['grid'][:, :, 0]
            Y[:] = f['grid'][:, :, 1]
            if adjust_map:
                self.x_min = np.min(X)
                self.x_max = np.max(X)
                self.y_min = np.min(Y)
                self.y_max = np.max(Y)
                self.setup_map()
            alpha_bg = 1.0

            # Resize window

            # X, Y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.nx_wind),
            #                    np.linspace(self.y_min, self.y_max, self.ny_wind))
            #
            # cartesian = np.dstack((X, Y)).reshape(-1, 2)
            #
            # uv = np.array(list(map(self.windfield, list(cartesian))))
            U = f['data'][0, :, :, 0].flatten()
            V = f['data'][0, :, :, 1].flatten()

            norms3d = np.sqrt(f['data'][0, :, :, 0] ** 2 + f['data'][0, :, :, 1] ** 2)
            norms = np.sqrt(U ** 2 + V ** 2)
            lognorms = np.log(np.sqrt(U ** 2 + V ** 2))

            norm = mpl_colors.Normalize()
            norm.autoscale(np.array([self.cm_norm_min, self.cm_norm_max]))

            sm = mpl_cm.ScalarMappable(cmap=self.cm, norm=norm)

            cb = self.map.colorbar(sm)  # , orientation='vertical')
            cb.set_label('Wind [m/s]')

            # Wind norm plot
            # znorms3d = scipy.ndimage.zoom(norms3d, 3)
            # zX = scipy.ndimage.zoom(X, 3)
            # zY = scipy.ndimage.zoom(Y, 3)
            znorms3d = norms3d
            zX = X
            zY = Y
            kwargs = {
                'cmap': self.cm,
                'norm': norm,
                'alpha': alpha_bg,
                'zorder': ZO_WIND_NORM,
                'shading': 'auto',
            }
            if self.coords == 'gcs':
                kwargs['latlon'] = True

            if wind_nointerp:
                self.map.pcolormesh(zX, zY, znorms3d, **kwargs)
            else:
                znorms3d = scipy.ndimage.zoom(norms3d, 3)
                zX = scipy.ndimage.zoom(X, 3)
                zY = scipy.ndimage.zoom(Y, 3)
                kwargs['antialiased'] = True
                kwargs['levels'] = 50
                self.map.contourf(zX, zY, znorms3d, **kwargs)

            # Quiver plot
            kwargs = {
                'color': (0.4, 0.4, 0.4, 1.0),
                'width': 0.001,
                'pivot': 'mid',
                'alpha': 0.7,
                'zorder': ZO_WIND_VECTORS
            }
            if self.coords == 'gcs':
                kwargs['latlon'] = True
            self.map.quiver(X, Y, U / norms, V / norms, **kwargs)  # color=color)

            # Stream plot
            kwargs = {'linewidth': 0.3, 'color': (0., 0., 0., alpha_bg)}
            if self.coords == 'cartesian':
                args = X[:, 0], Y[0, :], (f['data'][0, :, :, 0] / norms3d).transpose(), \
                       (f['data'][0, :, :, 1] / norms3d).transpose()
            elif self.coords == 'gcs':
                args = X.transpose(), Y.transpose(), (f['data'][0, :, :, 0] / norms3d).transpose(), \
                       (f['data'][0, :, :, 1] / norms3d).transpose()
                # args = X.transpose(), Y.transpose(), (f['data'][0, :, :, 0] / norms3d).transpose(), \
                #        (f['data'][0, :, :, 1] / norms3d).transpose()
                kwargs['latlon'] = True
            # self.map.streamplot(*args, **kwargs)  # color=color)

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

    def plot_traj(self, points, controls, ts, type, last_index, interrupted, coords, color_mode="default", nt_tick=10,
                  max_time=None, **kwargs):
        """
        Plots the given trajectory according to selected display mode
        :param nt_tick: Total number of ticking points to display (including start and end)
        """
        if max_time is not None:
            duration = max_time
        else:
            duration = (ts[-1] - ts[0])
        t_tick = duration / (nt_tick - 1)
        if not self.display_setup:
            self.setup()

        # Lines
        kwargs = {'color': reachability_colors[type]['steps'],
                  'zorder': ZO_TRAJS}
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        self.map.plot(points[:last_index - 1, 0], points[:last_index - 1, 1], **kwargs)

        # Ticks
        ticking_points = np.zeros((nt_tick, 2))
        nt = ts.shape[0]
        delta_t = duration / (nt - 1)
        k = 0
        for j, t in enumerate(ts):
            if j >= last_index:
                break
            if abs(t - k * t_tick) < 1.05 * (delta_t / 2.):
                ticking_points[k, :] = points[j]
                k += 1
        kwargs = {'s': 10. if interrupted else 5.,
                  'c': [reachability_colors[type]["time-tick"]],
                  'marker': 'o',
                  'linewidths': 1.,
                  'zorder': ZO_TRAJS}
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        self.map.scatter(ticking_points[1:k, 0], ticking_points[1:k, 1], **kwargs)

        # Last points
        kwargs = {'s': 10. if interrupted else 5.,
                  'c': [reachability_colors[type]["last"]],
                  'marker': (r'x' if interrupted else 'o'),
                  'linewidths': 1.,
                  'zorder': ZO_TRAJS}
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        self.map.scatter(points[last_index - 1, 0], points[last_index - 1, 1], **kwargs)

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

    def draw_trajs(self, filename, nt_tick=None, max_time=None):
        with h5py.File(os.path.join(self.output_path, filename), 'r') as f:
            for traj in f.values():
                if traj.attrs['coords'] != self.coords:
                    print(f'Warning : traj coord type {traj.attrs["coords"]} differs from display mode {self.coords}')
                kwargs = {
                    'color_mode': 'reachability',
                }
                if nt_tick is not None:
                    kwargs['nt_tick'] = nt_tick
                if max_time is not None:
                    kwargs['max_time'] = max_time
                self.plot_traj(traj['data'],
                               traj['controls'],
                               traj['ts'],
                               traj.attrs['type'],
                               traj.attrs['last_index'],
                               traj.attrs['interrupted'],
                               traj.attrs['coords'],
                               **kwargs)

    def draw_rff(self, filename, timeindex=None, debug=False):
        kwargs = {
            'zorder': ZO_RFF,
            'cmap': 'brg',
            'antialiased': True
        }
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        with h5py.File(os.path.join(self.output_path, filename), 'r') as f:
            nt, nx, ny = f['data'].shape
            ts_list = [timeindex] if timeindex is not None else range(nt)
            for k in ts_list:
                args = (f['grid'][:, :, 0], f['grid'][:, :, 1], f['data'][k, :, :]) + (
                    ([-1000., 1000.],) if not debug else ())
                self.map.contourf(*args, **kwargs)

    def show_params(self, fname=None):
        fname = self.params_fname if fname is None else fname
        with open(os.path.join(self.output_path, fname), 'r') as f:
            params = json.load(f)

        ps = ParamsSummary(style=self.params_ss_fname)
        md = ps.process_params(params)
        path = os.path.join(self.output_path, 'params.html')
        with open(path, "w", encoding="utf-8", errors="xmlcharrefreplace") as output_file:
            output_file.write(md)
        shutil.copyfile(os.path.join(self.params_ss_path, self.params_ss_fname),
                        os.path.join(self.output_path, self.params_ss_fname))
        webbrowser.open(path)

    def set_output_path(self, path):
        self.output_path = path
