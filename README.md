# icCNN
This repo is a restructure and a fix of https://github.com/ada-shen/icCNN. That repo is also an implementation of https://arxiv.org/abs/2107.04474, which makes this repo also an implementation of https://arxiv.org/abs/2107.04474.

# How to use?
Very simple:
1. Edit the `config.yaml` config file to your desired configuration
2. (Optional) You probably have to implement your own dataloader, so this repo is designed to support that (example in config.yaml)
3. run `python main.py -config_path config.yaml`

and that's it.

## Dependencies
A list of all important dependencies can be found in `requirement.txt`.

Note that I am using python 3.9.12, in case you want to run this code.