import argparse
import os.path
import sys

from mermoz.post_processing import PostProcessing
from mdisplay.frontend import FrontendHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory planning display tool')
    kwstore_cost = {
        'action': 'store_const',
        'const': True,
        'default': False
    }
    parser.add_argument('-l', '--latest', help='Run most recent results', action='store_const',
                        const=True, default=False)
    parser.add_argument('-L', '--last', help='Run last opened results', action='store_const',
                        const=True, default=False)
    parser.add_argument('-p', '--postprocessing', help='Run post processing', action='store_const',
                        const=True, default=False)
    parser.add_argument('-m', '--movie', help='Produce movie with case', action='store_const',
                        const=True, default=False)
    parser.add_argument('--frames', help='Number of frames for movie', nargs=1, default=50)
    parser.add_argument('--fps', help='Framerate for movie', nargs=1, default=10)
    parser.add_argument('--flags', help='Flags for display', default='')
    args = parser.parse_args(sys.argv[1:])

    fh = FrontendHandler()
    fh.setup()
    fh.select_example(select_latest=args.latest, select_last=args.last)
    fh.run_frontend(block=not args.postprocessing,
                    movie=args.movie,
                    frames=args.frames,
                    fps=args.fps,
                    flags=args.flags)
    if args.postprocessing:
        pp = PostProcessing(fh.example_dir)
        pp.load()
        pp.stats()
