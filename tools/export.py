import numpy as np
import numpy.typing as npt
import pyvista as pv
import ubelt as ub
from icecream import ic


def make_poly_data(_points: npt.ArrayLike, _faces: npt.ArrayLike) -> pv.PolyData:
    points: npt.NDArray[np.floating] = np.array(_points)
    faces: npt.NDArray[np.integer] = np.array(_faces, int)
    ic(_faces, faces)
    cells: npt.NDArray[np.integer] = np.empty(
        (faces.shape[0], faces.shape[1] + 1), faces.dtype
    )
    cells[:, 0] = faces.shape[1]
    cells[:, 1:] = faces
    return pv.PolyData(points, cells)


def main() -> None:
    fpath: str = ub.grabdata(
        "https://github.com/liblaf/archive-sculptor/raw/main/model/paradict.npy",
        hash_prefix="33a6a796e5180c7d",
        hasher="sha512",
    )
    para: dict[str, npt.ArrayLike] = np.load(fpath, allow_pickle=True).item()
    face: pv.PolyData = make_poly_data(para["template_face"], para["facialmesh_face"])
    face.save("template/face.ply")
    skull: pv.PolyData = make_poly_data(para["template_skull"], para["skullmesh_face"])
    skull.save("template/skull.ply")


if __name__ == "__main__":
    main()
