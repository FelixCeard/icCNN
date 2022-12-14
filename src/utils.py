import importlib
import os
from datetime import datetime

from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

model_dict = {
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152
}


class ModelNotFound(Exception):
    def __init__(self, x):
        super().__init__(x)


def load_model(args):
    return model_dict.get(args["model"]['type'])(num_class=args["model"]['num_classes'])


def load_dataset(args):
    if "args" not in args:
        return get_obj_from_str(args["dataloader"])()
    return get_obj_from_str(args["dataloader"])(**args.get("args", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def validate_yaml_config(args):
    # orga stuff
    if "task" not in args:
        raise KeyError("Expected the config file to contain the type of task with at the key 'task'")
    if "device" not in args:
        raise KeyError("Expected the config file to contain the type of device with at the key 'device'")

    # model
    if "model" not in args:
        raise KeyError("Expected the key 'model' to be present")
    if "type" not in args["model"]:
        raise KeyError("Expected the key 'type' in the key 'model' to be indicate the type of model")
    if args["model"]['type'] not in model_dict.keys():
        raise ModelNotFound(
            f"The provided model ({args['model']['type']}) is not implemented, please use on of the following models: {model_dict.keys()}")
    if "num_classes" not in args["model"]:
        args["model"]['num_classes'] = 1
    if "pretrained_path" in args["model"]:
        if not os.path.exists(args["model"]['pretrained_path']):
            raise FileNotFoundError(
                f"Could not find pre-trained weights at location {args['model']['pretrained_path']}")

    # datasets
    if "dataset" not in args:
        raise KeyError("Expected a Dataset or Dataloader to be included")
    if "train" not in args['dataset']:
        raise KeyError("Expected a train Dataset to be provided")

    if "dataloader" not in args["dataset"]["train"]:
        raise KeyError(
            "No Dataloader path has been included for the train set. The GitHub repo has an example how such input would look.")

    if "test" in args['dataset']:
        if "dataloader" not in args["dataset"]["test"]:
            raise KeyError(
                "No Dataloader path has been included for the test set. The GitHub repo has an example how such input would look.")

    # optimizer
    if "optimizer" not in args:
        args['optimizer'] = {}
        args['optimizer']['lr'] = 0.00001

    # training
    if "epochs" not in args["dataset"]["train"]:
        raise KeyError("No epoch was provided.")

    for k in ['batch_size', 'shuffle', 'num_workers']:
        if 'test' in args['dataset']:
            if k not in args['dataset']['train']:
                raise KeyError(f"Expected key '{k}' to be present in dataset->test")
        if k not in args['dataset']['train']:
            raise KeyError(f"Expected key '{k}' to be present in dataset->train")

    if 'path_weight_save' not in args['dataset']['train']:
        raise KeyError(f"Expected key 'path_weight_save' to be present in dataset->train")
    if 'save_every' not in args['dataset']['train']:
        raise KeyError(f"Expected key 'save_every' to be present in dataset->train")

    now = datetime.now()  # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    weight_save_path = os.path.join(args['dataset']['train']['path_weight_save'], date_time)
    args['dataset']['train']['weight_save_path'] = weight_save_path
    os.makedirs(weight_save_path, exist_ok=False)