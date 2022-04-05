import colorsys
import json
import os
import sys
import webbrowser

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, CheckButtons
from mpl_toolkits.basemap import Basemap
import h5py
import scipy.ndimage
from math import floor, cos, sin
import time

from mdisplay.font_config import FontsizeConf
from mdisplay.geodata import GeoData
from mdisplay.misc import *

state_names = [r"$x\:[m]$", r"$y\:[m]$"]
control_names = [r"$u\:[rad]$"]


class Display:
    """
    Defines all the visualization functions for navigation problems
    """

    def __init__(self, coords='cartesian', mode='only-map', title='Title', projection='merc', nocontrols=False):
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
        self.mainax = None

        # The object that handles plot, scatter, contourf... whatever cartesian or gcs
        self.ax = None

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

        # Time window upper bound for trajectories
        self.max_time = None
        # Number of expected time major ticks on trajectories
        self.nt_tick = None
        self.t_tick = None

        self.output_dir = None
        self.params_fname = 'params.json'
        self.params_fname_formatted = 'params.html'

        self.wind_fname = 'wind.h5'
        self.trajs_fname = 'trajectories.h5'
        self.rff_fname = 'rff.h5'
        self.wind_fpath = None
        self.trajs_fpath = None
        self.rff_fpath = None

        self.traj_artists = []
        self.traj_lines = []
        self.traj_ticks = []
        self.traj_lp = []
        self.traj_controls = []

        self.rff_contours = []

        self.label_list = []

        self.nocontrols = nocontrols

        self.ax_rbutton = None
        self.ax_cbutton = None
        self.control_button = None

        self.geodata = GeoData()

    def setup(self, bl=None, tr=None, bl_off=None, tr_off=None, projection='merc'):

        self.projection = projection

        if bl is None:
            if self.output_dir is None:
                print('Output path not set')
                exit(1)
            with open(os.path.join(self.output_dir, self.params_fname), 'r') as f:
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
            bottom=0.15,
            left=0.075,
            right=0.93,
            hspace=0.155,
            wspace=0.13
        )
        self.mainfig.suptitle(self.title)

        self.mainax = self.mainfig.add_subplot()

        if self.mode == "only-map":
            # gs = GridSpec(1, 1, figure=self.mainfig)
            # self.map = self.mainfig.add_subplot(gs[0, 0])
            pass
        elif self.mode == "full":
            """
            In this mode, display a second figure with state and control evolution over time
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
            In this mode, display a third figure with adjoint state evolution over time
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

        self.ax_rbutton = self.mainfig.add_axes([0.44, 0.025, 0.08, 0.05])
        self.reload_button = Button(self.ax_rbutton, 'Reload', color='white',
                                    hovercolor='grey')
        self.reload_button.label.set_fontsize(fsc.button_fontsize)
        self.reload_button.on_clicked(self.reload)

        self.ax_cbutton = self.mainfig.add_axes([0.54, 0.025, 0.08, 0.05])
        self.control_button = CheckButtons(self.ax_cbutton, ['Controls'], [not self.nocontrols])
        self.control_button.labels[0].set_fontsize(fsc.button_fontsize)
        self.control_button.on_clicked(self.toggle_controls)

    def setup_cm(self):
        cm_values = CM_WINDY_TRUNCATED
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
        cartesian = (self.coords == 'cartesian')
        gcs = (self.coords == 'gcs')

        if gcs:
            kwargs = {
                'resolution': 'l',
                'projection': self.projection,
                'ax': self.mainax
            }
            # Don't plot coastal lines features less than 1000km^2
            # kwargs['area_thresh'] = (6400e3 * np.pi / 180) ** 2 * (self.x_max - self.x_min) * 0.5 * (
            #             self.y_min + self.y_max) / 1000.

            if self.projection == 'merc':
                kwargs['llcrnrlon'] = self.x_min - self.x_offset * (self.x_max - self.x_min)
                kwargs['llcrnrlat'] = self.y_min - self.y_offset * (self.y_max - self.y_min)
                kwargs['urcrnrlon'] = self.x_max + self.x_offset * (self.x_max - self.x_min)
                kwargs['urcrnrlat'] = self.y_max + self.y_offset * (self.y_max - self.y_min)

            elif self.projection == 'ortho':
                kwargs['lon_0'] = 0.5 * (self.x_min + self.x_max)
                kwargs['lat_0'] = 0.5 * (self.y_min + self.y_max)

            elif self.projection == 'lcc':
                kwargs['lon_0'] = 0.5 * (self.x_min + self.x_max)
                kwargs['lat_0'] = 0.5 * (self.y_min + self.y_max)
                kwargs['lat_1'] = self.y_max + self.y_offset * (self.y_max - self.y_min)
                kwargs['lat_2'] = self.y_min - self.y_offset * (self.y_max - self.y_min)
                kwargs['width'] = (1 + self.x_offset) * (self.x_max - self.x_min) / 180 * np.pi * EARTH_RADIUS
                kwargs['height'] = kwargs[
                    'width']  # 1.5* (1 + self.y_offset) * (self.y_max - self.y_min) / 180 * np.pi * 6400e3

            else:
                print(f'Projection type "{self.projection}" not handled yet', file=sys.stderr)
                exit(1)

            self.map = Basemap(**kwargs)

            # self.map.shadedrelief()
            self.map.drawcoastlines()
            self.map.fillcontinents()
            lw = 0.5
            dashes = (2, 2)
            # draw parallels
            lat_min = min(self.y_min, self.y_max)
            lat_max = max(self.y_min, self.y_max)
            n_lat = floor((lat_max - lat_min) / 10) + 2
            self.map.drawparallels(10. * (floor(lat_min / 10.) + np.arange(n_lat)), labels=[1, 0, 0, 0],
                                   linewidth=lw,
                                   dashes=dashes)
            # draw meridians
            lon_min = min(self.x_min, self.x_max)
            lon_max = max(self.x_min, self.x_max)
            n_lon = floor((lon_max - lon_min) / 10) + 2
            self.map.drawmeridians(10. * (floor(lon_min / 10.) + np.arange(n_lon)), labels=[1, 0, 0, 1],
                                   linewidth=lw,
                                   dashes=dashes)

        if cartesian:
            self.mainax.axhline(y=0, color='k', linewidth=0.5)
            self.mainax.axvline(x=0, color='k', linewidth=0.5)
            self.mainax.axvline(x=1., color='k', linewidth=0.5)
            if self.axes_equal:
                self.mainax.axis('equal')
            self.mainax.set_xlim(self.x_min - self.x_offset * (self.x_max - self.x_min),
                                 self.x_max + self.x_offset * (self.x_max - self.x_min))
            self.mainax.set_ylim(self.y_min - self.y_offset * (self.y_max - self.y_min),
                                 self.y_max + self.y_offset * (self.y_max - self.y_min))
            self.mainax.set_xlabel('$x$ [m]')
            self.mainax.set_ylabel('$y$ [m]')
            self.mainax.grid(visible=True, linestyle='-.', linewidth=0.5)
            self.mainax.tick_params(direction='in')

        if cartesian:
            self.ax = self.mainax
        if gcs:
            self.ax = self.map

    def setup_map_adj(self):
        """
        Sets the display of the map for the adjoint state
        """
        self.map_adjoint.set_xlim(-1.1, 1.1)
        self.map_adjoint.set_ylim(-1.1, 1.1)

        self.map_adjoint.set_xlabel(r'$p_x\;[s/m]$')
        self.map_adjoint.set_ylabel(r'$p_y\;[s/m]$')

        self.map_adjoint.grid(visible=True, linestyle='-.', linewidth=0.5)
        self.map_adjoint.tick_params(direction='in')
        if self.axes_equal:
            self.map_adjoint.axis('equal')

    def draw_point_by_name(self, name):
        if self.coords != 'gcs':
            print('"draw_point_by_name" only available for GCS coordinates')
            exit(1)
        loc = self.geodata.get_coords(name)
        self.map.scatter(loc[0], loc[1], s=8., color='red', marker='D', latlon=True, zorder=ZO_ANNOT)
        self.mainax.annotate(name, self.map(loc[0], loc[1]), (10, 10), textcoords='offset pixels', ha='center')

    def draw_point(self, x, y, label=None):
        kwargs = {
            's': 8.,
            'color': 'red',
            'marker': 'D',
            'zorder': ZO_ANNOT
        }
        if self.coords == 'gcs':
            kwargs['latlon'] = True
            pos_annot = self.map(x, y)
        else:
            pos_annot = (x, y)
        self.ax.scatter(x, y, **kwargs)
        if label is not None:
            self.ax.annotate(label, pos_annot, (10, 10), textcoords='offset pixels', ha='center')

    def draw_wind(self, filename=None, adjust_map=False, wind_nointerp=True):
        """
        :param filename: Specify filename if different from standard
        :param adjust_map: Adjust map boundaries to the wind
        :param wind_nointerp: Draw wind as piecewise constant (pcolormesh plot)
        """

        if self.wind_fpath is None:
            filename = self.wind_fname if filename is None else filename
            self.wind_fpath = os.path.join(self.output_dir, filename)

        # Wind is piecewise constant
        wind_nointerp = wind_nointerp

        with h5py.File(self.wind_fpath, 'r') as f:
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
            # To avoid divide by zero
            eps = (self.x_max - self.x_min) * 1e-6
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

            norms3d = np.sqrt(f['data'][0, :, :, 0] ** 2 + f['data'][0, :, :, 1] ** 2) + eps
            norms = np.sqrt(U ** 2 + V ** 2) + eps

            norm = mpl_colors.Normalize()
            norm.autoscale(np.array([self.cm_norm_min, self.cm_norm_max]))

            sm = mpl_cm.ScalarMappable(cmap=self.cm, norm=norm)

            if self.coords == 'gcs':
                cb = self.ax.colorbar(sm)  # , orientation='vertical')
                cb.set_label('Wind [m/s]')
            elif self.coords == 'cartesian':
                cb = self.mainfig.colorbar(sm, ax=self.ax)
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
                self.ax.pcolormesh(zX, zY, znorms3d, **kwargs)
            else:
                znorms3d = scipy.ndimage.zoom(norms3d, 3)
                zX = scipy.ndimage.zoom(X, 3)
                zY = scipy.ndimage.zoom(Y, 3)
                kwargs['antialiased'] = True
                kwargs['levels'] = 50
                self.ax.contourf(zX, zY, znorms3d, **kwargs)

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
            self.ax.quiver(X, Y, U / norms, V / norms, **kwargs)  # color=color)

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

    # def setup_components(self):
    #     for k, state_plot in enumerate(self.state):
    #         state_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
    #         state_plot.tick_params(direction='in')
    #         state_plot.yaxis.set_label_position("right")
    #         state_plot.yaxis.tick_right()
    #         state_plot.set_ylabel(state_names[k])
    #         plt.setp(state_plot.get_xticklabels(), visible=False)
    #
    #     for k, control_plot in enumerate(self.control):
    #         control_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
    #         control_plot.tick_params(direction='in')
    #         control_plot.yaxis.set_label_position("right")
    #         control_plot.yaxis.tick_right()
    #         control_plot.set_ylabel(control_names[k])
    #         # Last plot
    #         if k == len(self.control) - 1:
    #             control_plot.set_xlabel(r"$t\:[s]$")

    def plot_traj(self, points, controls, ts, type, last_index, interrupted, coords, label=0, color_mode="default",
                  nt_tick=13,
                  max_time=None, nolabels=False, **kwargs):
        """
        Plots the given trajectory according to selected display mode
        :param nt_tick: Total number of ticking points to display (including start and end)
        """
        duration = (ts[last_index - 1] - ts[0])
        self.t_tick = max_time / (nt_tick - 1)
        if not self.display_setup:
            self.setup()

        # Lines
        if nolabels:
            p_label = None
        else:
            p_label = f'{type} {label}'
            if p_label not in self.label_list:
                self.label_list.append(p_label)
            else:
                p_label = None

        ls = linestyle[label % len(linestyle)]
        kwargs = {'color': reachability_colors[type]['steps'],
                  'linestyle': ls,
                  'label': p_label,
                  'zorder': ZO_TRAJS}
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        self.traj_lines.append(self.ax.plot(points[:last_index - 1, 0], points[:last_index - 1, 1], **kwargs))

        # Ticks
        ticking_points = np.zeros((nt_tick, 2))
        ticking_controls = np.zeros((nt_tick, 2))
        nt = ts.shape[0]
        # delta_t = max_time / (nt - 1)
        k = 0
        for j, t in enumerate(ts):
            if j >= last_index:
                break
            # if abs(t - k * t_tick) < 1.05 * (delta_t / 2.):
            if t - k * self.t_tick > -1e-3:
                if k >= ticking_points.shape[0]:
                    # Reallocate ticking points on demand
                    n_p, n_d = ticking_points.shape
                    _save = np.zeros((n_p, n_d))
                    _save[:] = ticking_points
                    ticking_points = np.zeros((n_p * 10, n_d))
                    ticking_points[:n_p] = _save[:]

                    # Reallocate controls too
                    n_p, n_d = ticking_controls.shape
                    _save = np.zeros((n_p, n_d))
                    _save[:] = ticking_controls
                    ticking_controls = np.zeros((n_p * 10, n_d))
                    ticking_controls[:n_p] = _save[:]

                ticking_points[k, :] = points[j]

                if self.coords == 'cartesian':
                    ticking_controls[k] = np.array([cos(controls[j]), sin(controls[j])])
                elif self.coords == 'gcs':
                    ticking_controls[k] = np.array([sin(controls[j]), cos(controls[j])])
                k += 1
        kwargs = {'s': 5.,
                  'c': [reachability_colors[type]["time-tick"]],
                  'marker': 'o',
                  'linewidths': 1.,
                  'zorder': ZO_TRAJS}
        if self.coords == 'gcs':
            kwargs['latlon'] = True

        self.traj_ticks.append(self.ax.scatter(ticking_points[1:k, 0], ticking_points[1:k, 1], **kwargs))

        # Heading vectors
        factor = 1. if self.coords == 'cartesian' else EARTH_RADIUS / 180 * np.pi
        kwargs = {
            'color': (0.2, 0.2, 0.2, 1.0),
            'pivot': 'tail',
            'alpha': 1.0,
            'zorder': ZO_WIND_VECTORS
        }
        if self.coords == 'gcs':
            kwargs['latlon'] = True
            kwargs['width'] = factor ** 2 / 1000000
            kwargs['scale'] = 1 / factor
            kwargs['units'] = 'xy'
        elif self.coords == 'cartesian':
            kwargs['width'] = 1 / 500
            kwargs['scale'] = 50

        if not self.nocontrols:
            self.traj_controls.append(self.ax.quiver(ticking_points[1:k, 0],
                                                     ticking_points[1:k, 1],
                                                     ticking_controls[1:k, 0],
                                                     ticking_controls[1:k, 1], **kwargs))

        # Last points
        kwargs = {'s': 10. if interrupted else 5.,
                  'c': [reachability_colors[type]["last"]],
                  'marker': (r'x' if interrupted else 'o'),
                  'linewidths': 1.,
                  'zorder': ZO_TRAJS}
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        self.traj_lp.append(self.ax.scatter(points[last_index - 1, 0], points[last_index - 1, 1], **kwargs))

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

    def draw_trajs(self, filename=None, nolabels=False):

        if self.trajs_fpath is None:
            filename = self.trajs_fname if filename is None else filename
            self.trajs_fpath = os.path.join(self.output_dir, filename)

        with h5py.File(self.trajs_fpath, 'r') as f:
            for traj in f.values():
                if traj.attrs['coords'] != self.coords:
                    print(f'Warning : traj coord type {traj.attrs["coords"]} differs from display mode {self.coords}')
                kwargs = {
                    'color_mode': 'reachability',
                }
                if self.nt_tick is not None:
                    kwargs['nt_tick'] = self.nt_tick
                if self.max_time is not None:
                    kwargs['max_time'] = self.max_time

                try:
                    label = int(traj.attrs['label'])
                except KeyError:
                    label = 0

                self.plot_traj(traj['data'],
                               traj['controls'],
                               traj['ts'],
                               traj.attrs['type'],
                               traj.attrs['last_index'],
                               traj.attrs['interrupted'],
                               traj.attrs['coords'],
                               label=label,
                               nolabels=nolabels,
                               **kwargs)

    def draw_rff(self, filename=None, timeindex=None, debug=False):
        if self.rff_fpath is None:
            filename = self.rff_fname if filename is None else filename
            self.rff_fpath = os.path.join(self.output_dir, filename)
        if not os.path.exists(self.rff_fpath):
            print(f'Failed to load RFF : File not found "{self.rff_fpath}"', file=sys.stderr)
            return
        with h5py.File(self.rff_fpath, 'r') as f:
            kwargs = {
                'zorder': ZO_RFF,
                'cmap': 'brg',
                'antialiased': True
            }
            if self.coords == 'gcs':
                kwargs['latlon'] = True
            nt, nx, ny = f['data'].shape
            ts_list = [timeindex] if timeindex is not None else range(nt)
            for k in ts_list:
                args = (f['grid'][:, :, 0], f['grid'][:, :, 1], f['data'][k, :, :]) + (
                    ([-1000., 1000.],) if not debug else ())
                self.rff_contours.append(self.ax.contourf(*args, **kwargs))

    def load_params(self, fname=None):
        """
        Load necessary data from parameter file.
        Currently only loads time window upper bound and trajectory ticking option
        :param fname: The parameters filename if different from standard
        """
        fname = self.params_fname if fname is None else fname
        with open(os.path.join(self.output_dir, fname), 'r') as f:
            params = json.load(f)
        try:
            self.nt_tick = params['nt_rft']
        except KeyError:
            pass
        try:
            self.max_time = params['max_time']
        except KeyError:
            pass

    def show_params(self, fname=None):
        fname = self.params_fname_formatted if fname is None else fname
        path = os.path.join(self.output_dir, fname)
        if not os.path.isfile(path):
            print('Parameters HTML file not found', file=sys.stderr)
        else:
            webbrowser.open(path)

    def set_output_path(self, path):
        self.output_dir = path

    def set_title(self, title):
        self.title = title

    def set_coords(self, coords):
        self.coords = coords

    def reload(self, event):
        t_start = time.time()
        print('Reloading... ', end='')

        # Reload params
        self.load_params()

        # Reload trajs
        if len(self.traj_lines) != 0:
            for l in self.traj_lines:
                for a in l:
                    a.remove()
            for a in self.traj_lp:
                a.remove()
            for a in self.traj_ticks:
                a.remove()
            for a in self.traj_controls:
                a.remove()
            self.traj_lines = []
            self.traj_ticks = []
            self.traj_lp = []
            self.traj_controls = []
            self.draw_trajs()

        # Reload RFFs
        print(len(self.rff_contours))
        if len(self.rff_contours) != 0:
            for c in self.rff_contours:
                for coll in c.collections:
                    try:
                        plt.gca().collections.remove(coll)
                    except ValueError:
                        pass
            self.rff_contours = []
            self.draw_rff()

        # Redraw
        self.mainfig.canvas.draw()  # redraw the figure
        t_end = time.time()
        print(f'Done ({t_end - t_start:.3f}s)')

    def legend(self):
        self.mainfig.legend()

    def toggle_controls(self, _):
        self.nocontrols = not self.nocontrols
        if self.nocontrols:
            for a in self.traj_controls:
                a.remove()
            self.traj_controls = []
            self.mainfig.canvas.draw()  # redraw the figure

    def update_title(self):
        fmax_time = f'{self.max_time/3600:.1f}h' if self.max_time > 1800. else f'{self.max_time:.2E}'
        ft_tick = f'{self.t_tick/3600:.1f}h' if self.t_tick > 1800. else f'{self.t_tick:.2E}'
        self.title += f' (ticks : {ft_tick})'
        self.mainfig.suptitle(self.title)

    def show(self, noparams=False):
        if not noparams:
            self.show_params()
        plt.show()
