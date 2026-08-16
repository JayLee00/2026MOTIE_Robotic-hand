# AnyPlace IssacLab Pick and PLace Pipeline

## Running Pick and Place

### Installing Dependencies

To run the simulation, first install the following dependencies in the sequential order:
- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#)
- [CuRobo](https://curobo.org/get_started/1_install_instructions.html)

### Running the simulation

Each experiment has a seperate folder (noted as `${exp_folder}`) that contains all model files and a configuration file `exp_config.json`. For more information, see the example under /testdata.

First, do pre-processing, including extracting 3D meshes from the USD files and adding physics properties:
```sh
python isaac_lab_pick_place/data/preprocess_data.py --exp ${exp_folder}
```

Then, execute AnyGrasp to obtain the grasps for the object (for installation, check the readme under `grasp_detection`):
```sh
python grasp_detection/main.py --exp ${exp_folder}
```

Finally, to run simulation in Isaac Lab:
```sh
python isaac_lab_pick_place/main.py --exp ${exp_folder} --method $method [--num-envs ${num_envs}]
```
where `$method` refers to the algorithm generating placement poses (to replace in the `rel_transform_path` value in the configuratino file), and `${num_envs}` for the number of environments of placement in parallel (set to 100 by default).

More information on the arguments can be seen by executing the script with the argument `-h`.