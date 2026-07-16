from importlib import import_module
import torch
from torch import nn
from functools import partial

import numpy as np
import random


def get_io_dims(data_loader):
    """
    Returns the shape of the dataset for each item within an entry returned by the `data_loader`
    The DataLoader object must return either a namedtuple, dictionary or a plain tuple.
    If `data_loader` entry is a namedtuple or a dictionary, a dictionary with the same keys as the
    namedtuple/dict item is returned, where values are the shape of the entry. Otherwise, a tuple of
    shape information is returned.

    Note that the first dimension is always the batch dim with size depending on the data_loader configuration.

    Args:
        data_loader (torch.DataLoader): is expected to be a pytorch Dataloader object returning
            either a namedtuple, dictionary, or a plain tuple.
    Returns:
        dict or tuple: If data_loader element is either namedtuple or dictionary, a ditionary
            of shape information, keyed for each entry of dataset is returned. Otherwise, a tuple
            of shape information is returned. The first dimension is always the batch dim
            with size depending on the data_loader configuration.
    """
    items = next(iter(data_loader))
    if hasattr(items, "_asdict"):  # if it's a named tuple
        items = items._asdict()

    if hasattr(items, "items"):  # if dict like
        return {k: v.shape for k, v in items.items()}
    else:
        return (v.shape for v in items)


def get_dims_for_loader_dict(dataloaders):
    """
    Given a dictionary of DataLoaders, returns a dictionary with same keys as the
    input and shape information (as returned by `get_io_dims`) on each keyed DataLoader.

    Args:
        dataloaders (dict of DataLoader): Dictionary of dataloaders.

    Returns:
        dict: A dict containing the result of calling `get_io_dims` for each entry of the input dict
    """
    return {k: get_io_dims(v) for k, v in dataloaders.items()}


def set_random_seed(seed: int, deterministic: bool = True):
    """
    Set random generator seed for Python interpreter, NumPy and PyTorch. When setting the seed for PyTorch,
    if CUDA device is available, manual seed for CUDA will also be set. Finally, if `deterministic=True`,
    and CUDA device is available, PyTorch CUDNN backend will be configured to `benchmark=False` and `deterministic=True`
    to yield as deterministic result as possible. For more details, refer to
    PyTorch documentation on reproducibility: https://pytorch.org/docs/stable/notes/randomness.html

    Beware that the seed setting is a "best effort" towards deterministic run. However, as detailed in the above documentation,
    there are certain PyTorch CUDA opertaions that are inherently non-deterministic, and there is no simple way to control for them.
    Thus, it is best to assume that when CUDA is utilized, operation of the PyTorch module will not be deterministic and thus
    not completely reproducible.

    Args:
        seed (int): seed value to be set
        deterministic (bool, optional): If True, CUDNN backend (if available) is set to be deterministic. Defaults to True. Note that if set
            to False, the CUDNN properties remain untouched and it NOT explicitly set to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)  # this sets both CPU and CUDA seeds for PyTorch


def split_module_name(abs_class_name):
    abs_module_path = ".".join(abs_class_name.split(".")[:-1])
    class_name = abs_class_name.split(".")[-1]
    return (abs_module_path, class_name)


def dynamic_import(abs_module_path, class_name):
    module_object = import_module(abs_module_path)
    target_class = getattr(module_object, class_name)
    return target_class


def resolve_fn(fn_name, default_base):
    """
    Given a string `fn_name`, resolves the name into a callable object. If the name has multiple `.` separated parts, treat all but the last
    as module names to trace down to the final name. If just the name is given, tries to resolve the name in the `default_base` module name context
    with direct eval of `{default_base}.{fn_name}` in this function's context.

    Raises `NameError` if no object matching the name is found and `TypeError` if the resolved object is not callabe.

    When successful, returns the resolved, callable object.
    """
    module_path, class_name = split_module_name(fn_name)

    try:
        fn_obj = (
            dynamic_import(module_path, class_name) if module_path else eval("{}.{}".format(default_base, class_name))
        )
    except NameError:
        raise NameError("Function `{}` does not exist".format(class_name))

    if not callable(fn_obj):
        raise TypeError("The object named {} is not callable.".format(class_name))

    return fn_obj


resolve_data = partial(resolve_fn, default_base="datasets")


def get_data(dataset_fn, dataset_config):
    """
    Resolves `dataset_fn` and invokes the resolved function onto the `dataset_config` configuration dictionary. The resulting
    dataloader will be returned.

    Args:
        dataset_fn: string name of the dataloader function path to be resolved. Alternatively, you can pass in a callable object and no name resolution will be performed.
        dataset_config: a dictionary containing keyword arguments to be passed into the resolved `dataset_fn`

    Returns:
        Result of invoking the resolved `dataset_fn` with `dataset_config` as keyword arguments.
    """
    if isinstance(dataset_fn, str):
        dataset_fn = resolve_data(dataset_fn)

    return dataset_fn(**dataset_config)
