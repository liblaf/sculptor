import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf import sculptor


def make_poly_data(points: ArrayLike, faces: ArrayLike) -> pv.PolyData:
    points: Float[np.ndarray, "V 3"] = np.array(points)
    faces: Integer[np.ndarray, "C 3|4"] = np.array(faces, int)
    return pv.PolyData.from_regular_faces(points, faces)


def main() -> None:
    para: sculptor.ParaDict = sculptor.load_paradict()
    face: pv.PolyData = make_poly_data(para["template_face"], para["facialmesh_face"])
    face.save("template/face.stl")
    skull: pv.PolyData = make_poly_data(para["template_skull"], para["skullmesh_face"])
    skull.save("template/skull.stl")


if __name__ == "__main__":
    main()
