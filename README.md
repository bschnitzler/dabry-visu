## Trajectory planning visualization tool

### Description

This tool is a companion to the `mermoz` module (<https://github.com/bschnitzler/mermoz>)

It handles the visualisation of trajectory optimization problems 
and the different objects at stake:
- Individual trajectories
- Wind fields
- Extremal fields
- Reachability fronts

It can handle time-varying winds, 2D planar as well as spherical (Earth)
problems.

### Getting started

1) Write the path to the `mermoz` module in the `MERMOZ_PATH` file.
2) Run the bash script `run.sh` (you may need to make it executable
before)

### Examples

Here are some previews of the tool

- Resolution with extremal field

![](https://github.com/bschnitzler/mdisplay/blob/main/res/eft.gif)

- Resolution with front tracking (interface to ToolboxLS)

![](https://github.com/bschnitzler/mdisplay/blob/main/res/rft.gif)