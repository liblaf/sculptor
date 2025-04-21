import numpy as np
import pyvista as pv
from jaxtyping import Float
from numpy.typing import ArrayLike

from ._data import ParaDict, load_paradict


def create_mesh_from_points(
    points: Float[ArrayLike, "V 3"],
) -> tuple[pv.PolyData, pv.PolyData]:
    points: Float[np.ndarray, "V 3"] = np.asarray(points)
    para: ParaDict = load_paradict()
    skull_n_points: int = para["template_skull"].shape[0]
    skull_points: Float[np.ndarray, "V 3"] = points[:skull_n_points]
    skull: pv.PolyData = pv.PolyData.from_regular_faces(
        skull_points, para["skullmesh_face"]
    )
    face_points: Float[np.ndarray, "V 3"] = points[skull_n_points:]
    face: pv.PolyData = pv.PolyData.from_regular_faces(
        face_points, para["facialmesh_face"]
    )
    return skull, face
