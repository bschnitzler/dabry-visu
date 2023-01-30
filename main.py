import argparse
import sys

from dabryvisu.frontend import FrontendHandler
from dabry.post_processing import PostProcessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory planning display tool')
    kwstore_cost = {
        'action': 'store_const',
        'const': True,
        'default': False
    }
    parser.add_argument('path', help='Path to source for display', nargs='?')
    parser.add_argument('-l', '--latest', help='Run most recent results', action='store_const',
                        const=True, default=False)
    parser.add_argument('-L', '--last', help='Run last opened results', action='store_const',
                        const=True, default=False)
    parser.add_argument('-p', '--postprocessing', help='Run post processing', action='store_const',
                        const=True, default=False)
    parser.add_argument('-m', '--movie', help='Produce movie with case', action='store_const',
                        const=True, default=False)
    parser.add_argument('--frames', help='Number of frames for movie', default=50)
    parser.add_argument('--fps', help='Framerate for movie', default=10)
    parser.add_argument('--flags', help='Flags for display', default='')
    args = parser.parse_args(sys.argv[1:])

    if args.path is not None:
        fh = FrontendHandler(mode='user')
        fh.setup()
        fh.example_dir = args.path
    else:
        fh = FrontendHandler()
        fh.setup()
        fh.select_example(select_latest=args.latest, select_last=args.last)
    fh.run_frontend(block=not args.postprocessing,
                    movie=args.movie,
                    frames=int(args.frames),
                    fps=int(args.fps),
                    flags=args.flags)
    if args.postprocessing:
        pp = PostProcessing(fh.example_dir)
        pp.load()
        pp.stats()
