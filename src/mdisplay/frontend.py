import os.path
import easygui
import h5py
import json

from mdisplay.display import Display
from mermoz.misc import *


class FrontendHandler:

    def __init__(self, mode='default'):
        self.display = None
        self.dd = None
        self.output_dir = None
        self.example_dir = None
        self.mode = mode
        self.pp_params = {}
        self.traj_stats = []

    def setup(self):
        base_path = os.environ.get('MERMOZ_PATH')
        if base_path is None:
            raise Exception('No path to Mermoz module. Please set environment variable MERMOZ_PATH and retry.')
        self.output_dir = os.path.join(base_path, 'output')

    def configure(self):
        self.display.set_title(os.path.basename(self.example_dir))
        self.display.import_params()
        self.display.load_all()
        self.display.setup()
        self.display.draw_all()

    def select_example(self, *args, select_latest=False, select_last=False):
        """
        Prompts user case selection depending on context. Sets cache accordingly
        """
        if self.mode == 'user':
            self.example_dir = args[0]
        elif self.mode == 'notebook':
            from ipywidgets import Dropdown
            from IPython.core.display import display
            cache_fp = os.path.join(self.output_dir, '.cache_frontend')
            default_value = None
            if os.path.exists(cache_fp):
                with open(cache_fp, 'r') as f:
                    default_value = f.readline()
            option_list = sorted(os.listdir(self.output_dir))
            new_ol = []
            for e in option_list:
                if not e.startswith('example_'):
                    del e
                else:
                    new_ol.append(e.split('example_')[1])
            kwargs = {'description': "Choose one:",
                      'options': new_ol}
            if default_value is not None and os.path.exists(os.path.join(self.output_dir, default_value)):
                kwargs['value'] = default_value
            self.dd = Dropdown(**kwargs)

            def handler(change):
                try:
                    with open(cache_fp, 'w') as f:
                        f.writelines(change['new'])
                except KeyError:
                    pass

            self.dd.observe(handler, names='value')
            display(self.dd)
        else:
            class ns:
                def __init__(self, value):
                    self.__dict__.update(value=value)

            cache_fp = os.path.join(self.output_dir, '.cache_frontend')
            last = '{LAST}'
            latest = '{LATEST}'
            nlist = [dd for dd in os.listdir(self.output_dir) if
                     os.path.isdir(os.path.join(self.output_dir, dd)) and not dd.startswith('.')]
            if select_latest:
                name = latest
            elif select_last:
                name = last
            else:
                name = easygui.choicebox('Choose example', 'Example1', [latest, last] + list(sorted(nlist)))
            if name is None:
                print('No example selected, quitting', file=sys.stderr)
                exit(1)
            sel_dir = os.path.join(self.output_dir, name)
            if name == latest:
                print('Opening latest file')
                all_subdirs = list(map(lambda n: os.path.join(self.output_dir, n), nlist))
                all_params = []
                for dd in all_subdirs:
                    params_path = os.path.join(dd, 'params.json')
                    if os.path.exists(params_path):
                        all_params.append(params_path)

                latest_subdir = os.path.dirname(max(all_params, key=os.path.getmtime))
                print(latest_subdir)
                self.dd = ns(os.path.basename(latest_subdir))
            elif name == last:
                print('Opening last file')
                with open(cache_fp, 'r') as f:
                    sel_dir = f.readline()
                    print(sel_dir)
                self.dd = ns(os.path.basename(sel_dir))
            else:
                # self.dd = ns(os.path.basename(sel_dir).split('example_')[1])
                self.dd = ns(os.path.basename(sel_dir))
                with open(cache_fp, 'w') as f:
                    f.writelines(sel_dir)

    def run_frontend(self, noparams=True, noshow=False, block=False, movie=False, frames=None, fps=None,
                     flags=''):
        if self.mode in ['notebook', 'default']:
            self.example_dir = os.path.join(self.output_dir, self.example_name())
        elif self.mode == 'user':
            pass
        else:
            print(f'Unknown mode {self.mode}', file=sys.stderr)
            exit(1)

        self.display = Display()
        self.display.set_output_path(self.example_dir)
        self.configure()
        self.display.set_mode(flags)
        self.display.draw_all()
        self.display.update_title()
        if movie:
            kwargs = {}
            if frames is not None:
                kwargs['frames'] = frames
            if fps is not None:
                kwargs['fps'] = fps
            self.display.to_movie(**kwargs)
        elif not noshow:
            try:
                self.display.show(noparams=noparams, block=block)
            except KeyboardInterrupt:
                pass

    def show_params(self):
        from IPython.core.display import HTML
        return HTML(filename=os.path.join(self.output_dir, self.example_name(), 'params.html'))

    def example_name(self):
        return ('example_' if self.mode == 'notebook' else '') + f'{self.dd.value}'

    def post_processing(self):
        with open(os.path.join(self.output_dir, self.example_name(), 'params.json'), 'r') as f:
            params = json.load(f)

        opti_ceil = params['target_radius']
        # print(opti_ceil)

        factor = DEG_TO_RAD if params['coords'] == COORD_GCS else 1.
        target = factor * np.array(params['point_target'])
        # print(target)

        reach_time_nowind = params['geodesic_time']

        reach_time_pmp = float('inf')
        length_pmp = float('inf')
        reach_time_int = []
        length_int = []

        with h5py.File(os.path.join(self.output_dir, self.example_name(), 'trajectories.h5')) as f:
            for i, traj in enumerate(f.values()):
                coords = traj.attrs['coords']
                last_index = traj.attrs['last_index']
                reach_time = float('inf')
                notfound = True
                last_p = np.zeros(2)
                length = 0.
                for k, p in enumerate(traj['data']):
                    if k >= last_index:
                        break
                    p = factor * np.array(p)
                    if k > 0:
                        length += distance(p, last_p, coords=coords)
                    if notfound and distance(p, target, coords=coords) < 1.05 * opti_ceil:
                        reach_time = traj["ts"][k]
                        notfound = False
                    last_p[:] = p
                if traj.attrs['type'] in ['pmp', 'optimal']:
                    # If traj is an extremal, only look for the best one
                    reach_time_pmp = min(reach_time_pmp, reach_time)
                    length_pmp = min(length_pmp, length)
                else:
                    # If traj is a regular integral, add it straightforward
                    tst = {
                        'id': k,
                        'name': 'integral',
                        'duration': reach_time,
                        'length': length
                    }
                    self.traj_stats.append(tst)

        # Add best extremal
        tst = {
            'id': -1,
            'name': 'best extremal',
            'duration': reach_time_pmp,
            'length': length_pmp
        }
        self.traj_stats.append(tst)

        # Add the no-wind geodesic
        tst = {
            'id': -2,
            'name': 'no-wind geodesic',
            'duration': params['geodesic_time'],
            'length': 1.,
        }
        try:
            tst['length'] = params['geodesic_length']
        except KeyError:
            tst['length'] = reach_time_nowind / params['airspeed']
        self.traj_stats.append(tst)

        self.pp_params['reach_time_geodesic'] = reach_time_nowind
        self.pp_params['reach_time_pmp'] = reach_time_pmp
        self.pp_params['length_pmp'] = length_pmp
        for i, rti in enumerate(reach_time_int):
            self.pp_params[f'reach_time_int_{i}'] = rti
            self.pp_params[f'length_int_{i}'] = length_int[i]

    def show_pp(self):
        def ft(value):
            # value is expected in seconds
            if value > 4000.:
                return f'{value / 3600.:.2f} h'
            return f'{value:.2f} s'

        def fl(value):
            # value expected in meters
            if value > 1000.:
                return f'{value / 1000.:.2f} km'
            return f'{value:.2f} m'

        s = "| Id | Name | Duration | Length | Mean groundspeed | \n|---|:---:|---|---|---|\n"
        for tst in self.traj_stats:
            s += f"| {tst['id']} | {tst['name']} | {ft(tst['duration'])} | {fl(tst['length'])} | {3.6 * tst['length'] / tst['duration']:.2f} km/h |\n"

        from IPython.core.display import display, Markdown
        # display(HTML("<style>.rendered_html { font-size: 30px; }</style>"))
        display(Markdown(s))
