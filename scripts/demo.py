import polyscope as ps
import polyscope.imgui as psim
import torch
from jaxtyping import Float

from liblaf import sculptor


class App:
    model: sculptor.SCULPTOR_layer
    jaw: Float[torch.Tensor, "1 3 1"]
    pose: Float[torch.Tensor, "1 6"]
    shape: Float[torch.Tensor, "1 60"]

    def __init__(self) -> None:
        ps.init()
        self.model = sculptor.SCULPTOR_layer()
        self.shape = torch.zeros((1, 60))
        self.pose = torch.zeros((1, 6))
        self.jaw = torch.zeros((1, 3, 1))
        ps.set_user_callback(self.callback)

    @property
    def skull_n_points(self) -> int:
        return self.model.template_skull.shape[0]

    def callback(self) -> None:
        jaw: Float[list[float], " 3"]
        _, jaw = psim.SliderFloat3("jaw", list(self.jaw[0, :, 0]), -20.0, 20.0)
        self.jaw[0, :, 0] = torch.as_tensor(jaw)
        _, pose = psim.SliderFloat3(
            "pose", list(torch.rad2deg(self.pose[0, 3:])), -45, 45
        )
        self.pose[0, 3:] = torch.deg2rad(torch.as_tensor(pose))
        self.update()
        _, shape = psim.SliderFloat4("shape[0:4]", list(self.shape[0, 0:4]), -3.0, 3.0)
        self.shape[0, 0:4] = torch.as_tensor(shape)

    def init_plot(self) -> None:
        ps.register_surface_mesh(
            "skull", self.model.template_skull, self.model.skullmesh_face
        )
        ps.register_surface_mesh(
            "face", self.model.template_face, self.model.facialmesh_face
        )

    def update(self) -> None:
        points: Float[torch.Tensor, "1 V 3"] = self.model(
            self.shape, self.pose, self.jaw
        )
        points: Float[torch.Tensor, "V 3"] = points.squeeze(0)
        skull: ps.SurfaceMesh = ps.get_surface_mesh("skull")
        skull.update_vertex_positions(points[: self.skull_n_points])
        face: ps.SurfaceMesh = ps.get_surface_mesh("face")
        face.update_vertex_positions(points[self.skull_n_points :])

    def show(self) -> None:
        ps.set_up_dir("z_up")
        ps.set_front_dir("neg_y_front")
        self.init_plot()
        ps.show()


app = App()
app.show()
