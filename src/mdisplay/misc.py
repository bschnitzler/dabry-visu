import numpy as np

my_red = np.array([0.8, 0., 0., 1.])
my_red_t = np.diag((1., 1., 1., 0.2)).dot(my_red)
my_orange = np.array([0.8, 0.5, 0., 1.])
my_orange_t = np.diag((1., 1., 1., 0.5)).dot(my_orange)
my_blue = np.array([0., 0., 0.8, 1.])
my_blue_t = np.diag((1., 1., 1., 0.5)).dot(my_blue)
my_black = np.array([0., 0., 0., 1.])
my_grey1 = np.array([0.75, 0.75, 0.75, 0.6])
my_grey2 = np.array([0.8, 0.8, 0.8, 0.6])
my_green = np.array([0., 0.8, 0., 1.])
my_green_t = np.diag((1., 1., 1., 0.5)).dot(my_green)

reachability_colors = {
    'pmp': {
        "steps": my_grey2,
        "time-tick": my_orange,
        "last": my_red
    },
    'integral': {
        "steps": my_grey1,
        "time-tick": my_orange,
        "last": my_blue
    },
    "approx": {
        "steps": my_grey1,
        "time-tick": my_orange_t,
        "last": my_orange
    },
    "point": {
        "steps": my_grey1,
        "time-tick": my_orange,
        "last": my_orange
    },
    "optimal": {
        "steps": my_green_t,
        "time-tick": my_green,
        "last": my_green
    },
}

monocolor_colors = {
    'pmp': my_red_t,
    'approx': my_orange_t,
    'point': my_blue,
    'integral': my_black
}