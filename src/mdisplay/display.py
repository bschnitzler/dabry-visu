import colorsys
import json
import os
import sys
import webbrowser
from datetime import datetime
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, CheckButtons, Slider
from mpl_toolkits.basemap import Basemap
import h5py
import scipy.ndimage
from math import floor, cos, sin
import time

from pyproj import Proj

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

        self.nt_wind = None

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
        self.sm = None

        self.cm_norm_min = None
        self.cm_norm_max = None

        # Time window upper bound for trajectories
        self.max_time = None
        # Number of expected time major ticks on trajectories
        self.nt_tick = None
        self.t_tick = None

        # Solver settings
        self.x_init = None
        self.x_target = None
        self.opti_ceil = None

        self.output_dir = None
        self.output_imgpath = None
        self.img_params = {
            'dpi': 300,
            'format': 'png'
        }
        self.params_fname = 'params.json'
        self.params_fname_formatted = 'params.html'

        self.wind_fname = 'wind.h5'
        self.trajs_fname = 'trajectories.h5'
        self.rff_fname = 'rff.h5'
        self.wind_fpath = None
        self.trajs_fpath = None
        self.rff_fpath = None

        self.wind = None
        self.trajs = []
        self.rff = None
        self.rff_cntr_kwargs = {
            'zorder': ZO_RFF,
            'cmap': 'brg',
            'antialiased': True,
            'alpha': .8
        }
        self.rff_zero_ceils = None
        self.nx_rft = None
        self.ny_rft = None
        self.nt_rft = None

        self.tv_wind = False

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
        self.ax_info = None

        self.wind_anchors = None
        self.wind_colormesh = None
        self.wind_colorcontour = None
        self.wind_quiver = None
        self.ax_timeslider = None
        self.time_slider = None
        self.wind_colorbar = None
        self.ax_timedisplay = None
        # Time window lower bound
        # Defined as wind time lower bound if wind is time-varying
        # Else minimum timestamps among trajectories and fronts
        self.tl = None
        # Time window upper bound
        # Maximum among all upper bounds (wind, trajs, rffs)
        self.tu = None
        # Current time step to display
        self.tcur = None

        self.color_mode = ''

        self.geodata = GeoData()

    def _index_wind(self):
        """
        Get nearest lowest index for time discrete grid (wind)
        :param t: The required time
        :return: Nearest lowest index for time, coefficient for the linear interpolation
        """
        nt = self.wind['ts'].shape[0]
        if self.tcur < self.wind['ts'][0]:
            return 0, 0.
        if self.tcur > self.wind['ts'][-1]:
            # Freeze wind to last frame
            return nt - 2, 1.
        tau = (self.tcur - self.tl) / (self.tu - self.tl)
        i, alpha = int(tau * (nt - 1)), tau * (nt - 1) - int(tau * (nt - 1))
        if i == nt - 1:
            i = nt - 2
            alpha = 1.
        return i, alpha


    def setup(self, bl=None, tr=None, bl_off=None, tr_off=None, projection='ortho', debug=False):
        self.projection = projection
        try:
            with open(os.path.join(self.output_dir, self.params_fname), 'r') as f:
                params = json.load(f)
                try:
                    self.nt_wind = params['nt_wind']
                except KeyError:
                    pass
        except FileNotFoundError:
            pass

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

        self.update_bounds((x_min, y_min), (x_max, y_max))

        # if self.opti_ceil is None:
        #     self.opti_ceil = 0.0005 * 0.5 * (self.x_max - self.x_min + self.y_max - self.y_min)

        if debug:
            print(self.x_min, self.x_max, self.y_min, self.y_max)

        fsc = FontsizeConf()
        plt.rc('font', size=fsc.fontsize)
        plt.rc('axes', titlesize=fsc.axes_titlesize)
        plt.rc('axes', labelsize=fsc.axes_labelsize)
        plt.rc('xtick', labelsize=fsc.xtick_labelsize)
        plt.rc('ytick', labelsize=fsc.ytick_labelsize)
        plt.rc('legend', fontsize=fsc.legend_fontsize)
        plt.rc('font', family=fsc.font_family)
        plt.rc('mathtext', fontset=fsc.mathtext_fontset)

        self.mainfig = plt.figure(num=f"Navigation problem ({self.coords})",
                                  constrained_layout=False,
                                  figsize=(12, 8))
        self.mainfig.subplots_adjust(
            top=0.93,
            bottom=0.17,
            left=0.15,
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

        self.ax_info = self.mainfig.text(0.34, 0.025, ' ')

        self.setup_slider()

    def setup_slider(self):
        self.ax_timeslider = self.mainfig.add_axes([0.03, 0.25, 0.0225, 0.63])
        self.ax_timedisplay = self.mainfig.text(0.03, 0.025, f'')
        val_init = 0.5
        self.time_slider = Slider(
            ax=self.ax_timeslider,
            label="Time",
            valmin=0.,
            valmax=1.,
            valinit=val_init,
            orientation="vertical"
        )
        self.reload_time(val_init)
        self.time_slider.on_changed(self.reload_time)

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
                proj = Proj(proj='ortho', lon_0=kwargs['lon_0'], lat_0=kwargs['lat_0'])

                kwargs['llcrnrx'], kwargs['llcrnry'] = proj(self.x_min - self.x_offset * (self.x_max - self.x_min),
                                                            self.y_min - self.y_offset * (self.y_max - self.y_min))
                kwargs['urcrnrx'], kwargs['urcrnry'] = proj(self.x_max + self.x_offset * (self.x_max - self.x_min),
                                                            self.y_max + self.y_offset * (self.y_max - self.y_min))

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

    def update_bounds(self, bl, tr):
        self.x_min = bl[0]
        self.y_min = bl[1]
        self.x_max = tr[0]
        self.y_max = tr[1]

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

    def clear_wind(self):
        if self.wind_colormesh is not None:
            self.wind_colormesh.remove()
            self.wind_colormesh = None
        if self.wind_colorcontour is not None:
            self.wind_colorcontour.remove()
            self.wind_colorcontour = None
        if self.wind_quiver is not None:
            self.wind_quiver.remove()
            self.wind_quiver = None
        if self.wind_anchors is not None:
            self.wind_anchors.remove()
            self.wind_anchors = None
        # if self.wind_colorbar is not None:
        #     self.wind_colorbar.remove()
        #     self.wind_colorbar = None

    def clear_trajs(self):
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

    def clear_rff(self):
        if len(self.rff_contours) != 0:
            print('cleaning fronts')
            for c in self.rff_contours:
                for coll in c.collections:
                    coll.remove()
            self.rff_contours = []

    def clear_solver(self):
        pass

    def load_wind(self, filename=None):
        self.wind = None
        if self.wind_fpath is None:
            filename = self.wind_fname if filename is None else filename
            self.wind_fpath = os.path.join(self.output_dir, filename)
        with h5py.File(self.wind_fpath, 'r') as f:
            self.wind = {}
            self.wind['data'] = np.zeros(f['data'].shape)
            self.wind['data'][:] = f['data']
            self.wind['attrs'] = {}
            for k, v in f.attrs.items():
                self.wind['attrs'][k] = v
            self.wind['grid'] = np.zeros(f['grid'].shape)
            self.wind['grid'][:] = f['grid']
            self.wind['ts'] = np.zeros(f['ts'].shape)
            self.wind['ts'][:] = f['ts']
            if self.wind['ts'].shape[0] > 1:
                self.tv_wind = True
                self.tl = self.wind['ts'][0]
                tu = self.wind['ts'][-1]
                if self.tu is None or tu > self.tu:
                    self.tu = tu

    def load_trajs(self, filename=None):
        self.trajs = []
        if self.trajs_fpath is None:
            filename = self.trajs_fname if filename is None else filename
            self.trajs_fpath = os.path.join(self.output_dir, filename)
        if not os.path.exists(self.trajs_fpath):
            print(f'[Warning] Trajectories not found at "{self.trajs_fpath}"', file=sys.stderr)
            return
        with h5py.File(self.trajs_fpath, 'r') as f:
            for k, traj in enumerate(f.values()):
                if traj.attrs['coords'] != self.coords:
                    print(
                        f'[Warning] Traj. coord type {traj.attrs["coords"]} differs from display mode {self.coords}')

                """
                if self.nt_tick is not None:
                    kwargs['nt_tick'] = self.nt_tick
                if self.max_time is not None:
                    kwargs['max_time'] = self.max_time
                """
                _traj = {}
                _traj['data'] = np.zeros(traj['data'].shape)
                _traj['data'][:] = traj['data']
                _traj['controls'] = np.zeros(traj['controls'].shape)
                _traj['controls'][:] = traj['controls']
                _traj['ts'] = np.zeros(traj['ts'].shape)
                _traj['ts'][:] = traj['ts']

                if not self.tv_wind:
                    tl = _traj['ts'][0]
                    if self.tl is None or tl < self.tl:
                        self.tl = tl
                tu = _traj['ts'][-1]
                if self.tu is None or tu > self.tu:
                    self.tu = tu

                _traj['type'] = traj.attrs['type']
                _traj['last_index'] = traj.attrs['last_index']
                _traj['interrupted'] = traj.attrs['interrupted']
                _traj['coords'] = traj.attrs['coords']
                _traj['label'] = traj.attrs['label']

                self.trajs.append(_traj)

    def load_rff(self, filename=None):
        self.rff = None
        self.rff_zero_ceils = []
        if self.rff_fpath is None:
            filename = self.rff_fname if filename is None else filename
            self.rff_fpath = os.path.join(self.output_dir, filename)
        if not os.path.exists(self.rff_fpath):
            Display._info(f'No RFF data found at "{self.rff_fpath}"')
            return
        with h5py.File(self.rff_fpath, 'r') as f:
            if self.coords != f.attrs['coords']:
                Display._warn(f'RFF coords "{f.attrs["coords"]}" does not match current display coords "{self.coords}"')

            if self.coords == 'gcs':
                self.rff_cntr_kwargs['latlon'] = True
            self.nt_rft, self.nx_rft, self.ny_rft = f['data'].shape
            # ceil = min((self.x_max - self.x_min) / (3 * self.nx_rft),
            #            (self.y_max - self.y_min) / (3 * self.ny_rft))
            nt = self.nt_rft
            self.rff_zero_ceils = []
            # if 'nt_rft_eff' in self.__dict__:
            #     nt = self.nt_rft_eff + 1
            # else:
            self.rff = {}
            failed_ceils = []
            self.rff = {}
            self.rff['data'] = np.zeros(f['data'].shape)
            self.rff['data'][:] = f['data']
            self.rff['grid'] = np.zeros(f['grid'].shape)
            self.rff['grid'][:] = f['grid']
            self.rff['ts'] = np.zeros(f['ts'].shape)
            self.rff['ts'][:] = f['ts']

            for k in range(nt):

                if not self.tv_wind:
                    tl = self.rff['ts'][0]
                    if self.tl is None or tl < self.tl:
                        self.tl = tl

                tu = self.rff['ts'][-1]
                if self.tu is None or tu > self.tu:
                    self.tu = tu

                data_max = self.rff['data'][k, :, :].max()
                data_min = self.rff['data'][k, :, :].min()

                zero_ceil = min((self.rff['grid'][:, :, 0].max() - self.rff['grid'][:, :, 0].min()) / (3 * self.nx_rft),
                           (self.rff['grid'][:, :, 1].max() - self.rff['grid'][:, :, 1].min()) / (3 * self.ny_rft))
                self.rff_zero_ceils.append(zero_ceil)

                if data_min > zero_ceil or data_max < -zero_ceil:
                    failed_ceils.append(k)

                # Adjust zero ceil if needed
                minabs = np.abs(self.rff['data']).min()
                if minabs > self.rff_zero_ceils[k] / 2:
                    Display._info('Detecting wrong zero_ceil, adjusting automatically')
                    self.rff_zero_ceils[k] = minabs * 1.1

            Display._warn(f'No RFF value in zero band for indexes {tuple(failed_ceils)}')

    def import_params(self, fname=None):
        """
        Load necessary data from parameter file.
        Currently only loads time window upper bound and trajectory ticking option
        :param fname: The parameters filename if different from standard
        """
        fname = self.params_fname if fname is None else fname
        with open(os.path.join(self.output_dir, fname), 'r') as f:
            params = json.load(f)
        try:
            self.coords = params['coords']
        except KeyError:
            pass
        try:
            self.nt_tick = params['nt_rft']
        except KeyError:
            pass
        try:
            self.max_time = params['max_time']
        except KeyError:
            pass
        try:
            self.t_tick = self.max_time / (self.nt_tick - 1)
        except TypeError:
            pass
        try:
            self.x_init = params['point_init']
        except KeyError:
            pass
        try:
            self.x_target = params['point_target']
        except KeyError:
            pass
        try:
            self.opti_ceil = params['target_radius']
        except KeyError:
            print('Failed to load "target_radius", using default value', file=sys.stderr)
        try:
            self.nt_rft_eff = params['nt_rft_eff']
        except KeyError:
            pass

    def load_all(self):
        self.load_wind()
        self.load_trajs()
        self.load_rff()
        n_trajs = len(self.trajs)
        n_rffs = 0 if self.rff is None else self.rff['data'].shape[0]
        self.tcur = 0.5 * (self.tl + self.tu)
        Display._info(f'Loading ended with {n_trajs} trajs and {n_rffs} RFFs.')

    def draw_rff(self, debug=False):
        self.clear_rff()
        if self.rff is not None:
            ax = None
            if debug:
                fig, ax = plt.subplots()
            nt = self.rff['data'].shape[0]

            il = 0
            iu = nt
            ts = self.rff['ts']
            at_least_one = False
            for i in range(ts.shape[0]):
                if ts[i] < self.tl:
                    il += 1
                elif ts[i] > self.tcur:
                    iu = i - 1
                    break
                else:
                    at_least_one = True
            if not at_least_one:
                return
            if iu <= il:
                return

            print(il, iu)

            for k in range(il, iu):
                data_max = self.rff['data'][k, :, :].max()
                data_min = self.rff['data'][k, :, :].min()
                if debug:
                    ax.hist(self.rff['data'][k, :, :].reshape(-1), density=True, label=k,
                            color=path_colors[k % len(path_colors)])
                zero_ceil = self.rff_zero_ceils[k]  # (data_max - data_min) / 1000.
                if debug:
                    print(f'{k}, min : {data_min}, max : {data_max}, zero_ceil : {zero_ceil}')
                args = (self.rff['grid'][:, :, 0], self.rff['grid'][:, :, 1], self.rff['data'][k, :, :]) + (
                    ([-zero_ceil / 2, zero_ceil / 2],) if not debug else ([data_min, 0., data_max],))
                # ([-1000., 1000.],) if not debug else (np.linspace(-100000, 100000, 200),))
                self.rff_contours.append(self.ax.contourf(*args, **self.rff_cntr_kwargs))
            if debug:
                ax.legend()
                plt.show()

    def draw_wind(self, adjust_map=False, wind_nointerp=None, autoscale=False, showanchors=False,
                  autoquiversample=False):
        """
        :param filename: Specify filename if different from standard
        :param adjust_map: Adjust map boundaries to the wind
        :param wind_nointerp: Draw wind as piecewise constant (pcolormesh plot)
        """

        # Erase previous drawings if existing
        self.clear_wind()

        nt, nx, ny, _ = self.wind['data'].shape
        X = np.zeros((nx, ny))
        Y = np.zeros((nx, ny))
        if autoquiversample:
            ur = nx // 40
        else:
            ur = 1
        X[:] = self.wind['grid'][:, :, 0]
        Y[:] = self.wind['grid'][:, :, 1]
        if adjust_map:
            self.update_bounds((np.min(X), np.min(Y)), (np.max(X), np.max(Y)))
            self.setup_map()

        alpha_bg = 1.0
        if wind_nointerp is None:
            try:
                # True if wind displaystyle is piecewise constant, else smooth
                wind_nointerp = not self.wind['attrs']['analytical']
            except KeyError:
                wind_nointerp = True

        # Resize window

        # X, Y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.nx_wind),
        #                    np.linspace(self.y_min, self.y_max, self.ny_wind))
        #
        # cartesian = np.dstack((X, Y)).reshape(-1, 2)
        #
        # uv = np.array(list(map(self.windfield, list(cartesian))))
        U_grid = np.zeros((nx, ny))
        V_grid = np.zeros((nx, ny))
        if not self.tv_wind:
            U_grid[:] = self.wind['data'][0, :, :, 0]
            V_grid[:] = self.wind['data'][0, :, :, 1]
        else:
            k, p = self._index_wind()
            U_grid[:] = (1 - p) * self.wind['data'][k, :, :, 0] + p * self.wind['data'][k + 1, :, :, 0]
            V_grid[:] = (1 - p) * self.wind['data'][k, :, :, 1] + p * self.wind['data'][k + 1, :, :, 1]
        U = U_grid.flatten()
        V = V_grid.flatten()

        norms3d = np.sqrt(U_grid ** 2 + V_grid ** 2)
        norms = np.sqrt(U ** 2 + V ** 2)
        eps = (np.max(norms) - np.min(norms)) * 1e-6
        # norms += eps

        norm = mpl_colors.Normalize()
        if autoscale:
            self.cm = 'Blues_r'
            norm.autoscale(norms)
        else:
            self.cm = windy_cm
            norm.autoscale(np.array([windy_cm.norm_min, windy_cm.norm_max]))

        if self.sm is None:
            self.sm = mpl_cm.ScalarMappable(cmap=self.cm, norm=norm)

            if self.coords == 'gcs':
                self.wind_colorbar = self.ax.colorbar(self.sm)  # , orientation='vertical')
                self.wind_colorbar.set_label('Wind [m/s]')
            elif self.coords == 'cartesian':
                self.wind_colorbar = self.mainfig.colorbar(self.sm, ax=self.ax)
                self.wind_colorbar.set_label('Wind [m/s]')

        # Wind anchor plot
        if showanchors:
            self.wind_anchors = self.ax.scatter(X, Y, zorder=ZO_WIND_ANCHORS)

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
        }
        if self.coords == 'gcs':
            kwargs['latlon'] = True

        if wind_nointerp:
            kwargs['shading'] = 'auto'
            self.wind_colormesh = self.ax.pcolormesh(zX, zY, znorms3d, **kwargs)
        else:
            znorms3d = scipy.ndimage.zoom(norms3d, 3)
            zX = scipy.ndimage.zoom(X, 3)
            zY = scipy.ndimage.zoom(Y, 3)
            kwargs['antialiased'] = True
            kwargs['levels'] = 50
            self.wind_colorcontour = self.ax.contourf(zX, zY, znorms3d, **kwargs)

        # Quiver plot
        qX = X[::ur, ::ur]
        qY = Y[::ur, ::ur]
        qU = U_grid[::ur, ::ur].flatten()
        qV = V_grid[::ur, ::ur].flatten()
        qnorms = 1e-6 + np.sqrt(qU ** 2 + qV ** 2)
        kwargs = {
            'color': (0.4, 0.4, 0.4, 1.0),
            'width': 0.001,
            'pivot': 'mid',
            'alpha': 0.7,
            'zorder': ZO_WIND_VECTORS
        }
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        self.wind_quiver = self.ax.quiver(qX, qY, qU / qnorms, qV / qnorms, **kwargs)  # color=color)

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

    def draw_trajs(self, nolabels=False, opti_only=False):
        self.clear_trajs()
        for k, traj in enumerate(self.trajs):
            if not opti_only or traj['type'] in ['optimal', 'integral']:
                self.draw_traj(k, nolabels=nolabels)

    def draw_traj(self, itr, nolabels=False):
        """
        Plots the given trajectory according to selected display mode
        """
        # duration = (ts[last_index - 1] - ts[0])
        # self.t_tick = max_time / (nt_tick - 1)
        if not self.display_setup:
            self.setup()

        points = self.trajs[itr]['data']
        controls = self.trajs[itr]['controls']
        ts = self.trajs[itr]['ts']
        last_index = self.trajs[itr]['last_index']
        label = self.trajs[itr]['label']
        idfr = label
        interrupted = self.trajs[itr]['interrupted']
        _type = self.trajs[itr]['type']

        # Lines
        if nolabels:
            p_label = None
        else:
            p_label = f'{_type} {label}'
            if p_label not in self.label_list:
                self.label_list.append(p_label)
            else:
                p_label = None

        # Determine range of indexes that can be plotted
        il = 0
        iu = last_index - 1

        at_least_one = False
        for i in range(ts.shape[0]):
            if ts[i] < self.tl:
                il += 1
            elif ts[i] > self.tcur:
                iu = i - 1
                break
            else:
                at_least_one = True
        if not at_least_one:
            return
        if iu <= il:
            return

        ls = linestyle[label % len(linestyle)]
        kwargs = {
            'color': reachability_colors[_type]['steps'] if _type != 'path' else path_colors[idfr % len(path_colors)],
            'linestyle': ls,
            'label': p_label,
            'gid': idfr,
            'zorder': ZO_TRAJS}
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        self.traj_lines.append(self.ax.plot(points[il:iu - 1, 0], points[il:iu - 1, 1], **kwargs))

        """
        # Ticks
        if self.t_tick is not None:
            ticking_points = np.zeros((nt_tick, 2))
            ticking_controls = np.zeros((nt_tick, 2))
            nt = ts.shape[0]
            # delta_t = max_time / (nt - 1)
            k = 0
            for j, t in enumerate(ts):
                if j >= last_index:
                    break
                # if abs(t - k * t_tick) < 1.05 * (delta_t / 2.):
                if t - k * self.t_tick > -1e-3 or j == 0:
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
        """

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

        """
        if not self.nocontrols:
            self.traj_controls.append(self.ax.quiver(ticking_points[1:k, 0],
                                                     ticking_points[1:k, 1],
                                                     ticking_controls[1:k, 0],
                                                     ticking_controls[1:k, 1], **kwargs))
        """
        # Last points
        kwargs = {'s': 10. if interrupted else 5.,
                  'c': [reachability_colors[_type]["last"]] if _type != 'path' else path_colors[
                      idfr % len(path_colors)],
                  'marker': (r'x' if interrupted else 'o'),
                  'linewidths': 1.,
                  'zorder': ZO_TRAJS}
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        self.traj_lp.append(self.ax.scatter(points[iu, 0], points[iu, 1], **kwargs))

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

    def draw_solver(self, labeling=True):
        kwargs = {}
        if self.coords == 'gcs':
            kwargs['latlon'] = True
            scatterax = self.map
        else:
            scatterax = self.mainax
        has_opti_ceil = self.opti_ceil is not None
        if not has_opti_ceil:
            print('Missing opti_ceil', file=sys.stderr)
        # Init point
        if self.x_init is not None:
            scatterax.scatter(self.x_init[0], self.x_init[1], s=8., color='blue', marker='D', zorder=ZO_ANNOT, **kwargs)
            c = self.map(self.x_init[0], self.x_init[1]) if self.coords == 'gcs' else (self.x_init[0], self.x_init[1])
            if labeling:
                self.mainax.annotate('Init', c, (10, 10), textcoords='offset pixels', ha='center')
            if has_opti_ceil:
                self.mainax.add_patch(plt.Circle(c, self.opti_ceil))
        else:
            print('Missing x_init', file=sys.stderr)
        # Target point
        if self.x_target is not None:
            scatterax.scatter(self.x_target[0], self.x_target[1], s=8., color='blue', marker='D', zorder=ZO_ANNOT,
                              **kwargs)
            c = self.map(self.x_target[0], self.x_target[1]) if self.coords == 'gcs' else (
                self.x_target[0], self.x_target[1])
            if labeling:
                self.mainax.annotate('Target', c, (10, 10), textcoords='offset pixels', ha='center')
            if has_opti_ceil:
                self.mainax.add_patch(plt.Circle(c, self.opti_ceil))
        else:
            print('Missing x_target', file=sys.stderr)

    def draw_all(self):
        self.draw_wind()
        self.draw_trajs()
        self.draw_rff()
        # self.draw_solver()

        self.mainfig.canvas.draw()

    def show_params(self, fname=None):
        fname = self.params_fname_formatted if fname is None else fname
        path = os.path.join(self.output_dir, fname)
        if not os.path.isfile(path):
            print('Parameters HTML file not found', file=sys.stderr)
        else:
            webbrowser.open(path)

    def set_output_path(self, path):
        self.output_dir = path
        basename = os.path.basename(path)
        self.output_imgpath = os.path.join(path, basename + f'.{self.img_params["format"]}')

    def set_title(self, title):
        self.title = title

    def set_coords(self, coords):
        self.coords = coords

    def reload(self, event):
        t_start = time.time()
        print('Reloading... ', end='')

        # Reload params
        self.import_params()
        self.load_all()

        self.draw_all()

        t_end = time.time()
        print(f'Done ({t_end - t_start:.3f}s)')

    def reload_time(self, val):
        self.tcur = self.tl + val * (self.tu - self.tl)
        d = datetime.fromtimestamp(self.tcur)
        self.ax_timedisplay.set_text(f'{str(d).split(".")[0]}')

        self.draw_all()

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
        try:
            fmax_time = f'{self.max_time / 3600:.1f}h' if self.max_time > 1800. else f'{self.max_time:.2E}'
            if self.t_tick is not None:
                ft_tick = f'{self.t_tick / 3600:.1f}h' if self.t_tick > 1800. else f'{self.t_tick:.2E}'
                self.title += f' (ticks : {ft_tick})'
            self.mainfig.suptitle(self.title)
        except TypeError:
            pass

    def show(self, noparams=False):
        if not noparams:
            self.show_params()
        self.mainfig.savefig(self.output_imgpath, **self.img_params)
        plt.show()

    @staticmethod
    def _warn(msg):
        print(f'[Warning] {msg}', file=sys.stderr)

    @staticmethod
    def _info(msg):
        print(f'[Info] {msg}')
