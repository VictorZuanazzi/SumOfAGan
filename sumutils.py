import os
from glob import glob
from os.path import join, getctime

from typing import NoReturn, Tuple, Dict

import torch


def save_top_k(state_dict: Dict, path: str, name: str, k: int) -> bool:
    """Before saving state dict removes all old files. Only the k most
    recent files will remain.

    Args:
        state_dict: the dictionary to be saved.
        path: path to folder.
        name: name of the file.
        k: number of files to be kept in the folder.

    Returns:
        True if saved successfully, False otherwise.
    """

    os.makedirs(path, exist_ok=True)
    saved_files = glob(join(path, "*"))
    saved_files.sort(key=getctime)
    delete_list = saved_files[:-k]

    for file in delete_list:
        os.remove(file)

    try:
        torch.save(state_dict, join(path, name + ".pt"))
        return True

    except OSError as err:
        print(f"Could not save state dict. Error raised: {err}.")
        return False



