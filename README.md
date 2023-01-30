## Trajectory planning visualization tool

### Description

This tool is a companion to the `dabry` module (<https://github.com/bschnitzler/dabry>)

It handles the visualisation of trajectory optimization problems 
and the different objects at stake:
- Individual trajectories
- Wind fields
- Extremal fields
- Reachability fronts

It can handle time-varying winds, 2D planar as well as spherical (Earth)
problems.

### Getting started

1) Write the path to the `dabry` module in the `DABRYPATH` file.
2) Run the script `run.sh` using for instance
```sh
bash run.sh
```

### Examples

Here are some previews of the tool

- Resolution with extremal field

![](https://github.com/bschnitzler/dabry-visu/blob/main/res/eft.gif)

- Resolution with front tracking (interface to ToolboxLS)

![](https://github.com/bschnitzler/dabry-visu/blob/main/res/rft.gif)