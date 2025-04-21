import functools
from typing import TypedDict

import numpy as np
import pooch
import torch
from jaxtyping import Float


class ParaDict(TypedDict):
    skullmesh_face: Float[np.ndarray, "54110 3"]
    facialmesh_face: Float[np.ndarray, "10606 4"]
    template_skull: Float[np.ndarray, "27051 3"]
    template_face: Float[np.ndarray, "10710 3"]
    skull_shape: Float[np.ndarray, "50 27051 3"]
    face_shape: Float[np.ndarray, "50 10710 3"]
    skull_div: Float[np.ndarray, "10 27051 3"]
    face_div: Float[np.ndarray, "10 10710 3"]
    template: Float[torch.Tensor, "37761 3"]
    shape_dirs: Float[torch.Tensor, "37761 3 60"]
    parents: Float[torch.Tensor, "2"]
    J_reg: Float[torch.Tensor, "2 37761"]
    pose_dir: Float[torch.Tensor, "9 113283"]
    lbs_weights: Float[torch.Tensor, "37761 3"]


@functools.cache
def load_paradict() -> ParaDict:
    paradict_path: str = pooch.retrieve(
        url="https://raw.githubusercontent.com/liblaf/archive-sculptor/main/model/paradict.npy",
        known_hash="sha256:43b0773ef04c20afd12789d68c492115d6ffe11038d600eb9b989a4af4c91f00",
    )
    return np.load(paradict_path, allow_pickle=True).item()
