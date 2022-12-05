import argparse
import os

import torch

from train import train_model

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from utils import validate_yaml_config

if __name__ == '__main__':
    torch.set_num_threads(5)

    # read config file
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', type=str, help='path to config file')
    input_args = parser.parse_args()

    if not os.path.exists(input_args.config_path):
        raise FileNotFoundError(f"Could not find config file located at {input_args.config_path}")

    # parse yaml config file
    with open(input_args.config_path, 'r') as file:
        content = file.read()
    args = load(content, Loader=Loader)

    validate_yaml_config(args)

    # switch to task-specific script
    if args['task'] == 'train':
        train_model(args)
    else:
        raise NotImplementedError("Currently, only train is implemented")