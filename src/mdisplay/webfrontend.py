import os
import mpld3
import shutil
import eel

from mdisplay.display import Display


class WebFrontend:

    def __init__(self):
        self.tmp_path = 'tmp'
        self.data_path = '.data'
        self.html_srcfile = 'main.html'
        self.html_src_fpath = os.path.join(self.data_path, self.html_srcfile)
        self.html_dst_fpath = os.path.join(self.tmp_path, self.html_srcfile)
        self.figure_fpath = os.path.join(self.tmp_path, 'figure.html')

    def run(self):
        if os.path.exists(self.tmp_path):
            shutil.rmtree(self.tmp_path)
        os.mkdir(self.tmp_path)
        shutil.copy(self.html_src_fpath, self.html_dst_fpath)

        output_path = '/home/bastien/Documents/work/mermoz/output/example_solver-ef_3obs'
        self.display = Display()
        self.display.set_output_path(output_path)
        self.display.nocontrols = True
        self.display.set_title(os.path.basename(output_path))
        self.display.import_params()
        self.display.load_trajs()
        self.display.setup()
        self.display.draw_wind()
        self.display.draw_trajs(nolabels=False)
        self.display.draw_rff()
        self.display.draw_solver()
        html_str = mpld3.fig_to_html(self.display.mainfig)
        with open(self.figure_fpath, "w") as f:
            f.write(html_str)

        eel.init(self.tmp_path)
        eel.start(self.html_srcfile, mode='mozilla')


if __name__ == '__main__':
    wf = WebFrontend()
    wf.run()
