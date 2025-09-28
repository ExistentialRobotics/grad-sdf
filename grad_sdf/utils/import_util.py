import argparse
from importlib import import_module


def get_dataset(dataset_name: str, dataset_args: dict):
    Dataset = import_module("grad_sdf.dataset." + dataset_name)
    return Dataset.DataLoader(**dataset_args)


def get_property(args, name, default):
    if isinstance(args, dict):
        return args.get(name, default)
    elif isinstance(args, argparse.Namespace):
        if hasattr(args, name):
            return vars(args)[name]
        else:
            return default
    else:
        raise ValueError(f"unkown dict/namespace type: {type(args)}")
