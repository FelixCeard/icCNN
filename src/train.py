from utils import load_model, get_obj_from_str, load_dataset


def train_model(args):

    # load the model
    model = load_model(args)
    if "pretrained_path" in args["model"]:
        model.load_state_dict(args["model"]['pretrained_path'])

    # load the dataset
    train_dataset = load_dataset(args["dataset"]["train"])
    test_dataset = load_dataset(args["dataset"]["test"])