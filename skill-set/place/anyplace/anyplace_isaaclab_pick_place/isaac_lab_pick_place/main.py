import os
import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="NVIDIA Isaac Lab Grasp Simulation")
parser.add_argument("--num-envs", type=int, default=100, help="the number of environments")
parser.add_argument("--method", type=str, required=True, help="the relative pose transform to be used")
parser.add_argument("--exp", type=str, required=True, help="the path to the experiment folder")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from experiment import PickPlaceExperiment


def main():
    exp_path = args_cli.exp
    exp_cfg_path = os.path.join(exp_path, "exp_config.json")

    pick_place_exp = PickPlaceExperiment(
        args_cli.method,
        exp_cfg_path,
        args_cli.num_envs,
        device=args_cli.device,
    )
    print(pick_place_exp.execute())

    pick_place_exp.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
