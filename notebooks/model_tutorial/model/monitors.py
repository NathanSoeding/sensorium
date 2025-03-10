import torch
from .utils import isotropic_grid_2d, rmat_3d
from .parameters import Parameter
from .modules import Module


# -------------- Monitor Base --------------


class Monitor(Module):
    """Monitor Module"""

    def project(self, rays):
        """
        Parameters
        ----------
        rays : Tensor
            [N, H, W, 3], grid of 3D rays

        Returns
        -------
        Tensor
            [N, H, W, 2], 2D projection
        """
        raise NotImplementedError()

    def rays(self, batch_size=1, height=144, width=256):
        """
        Parameters
        ----------
        batch_size : int
            batch size (N)
        height : int
            output height (H)
        width : int
            output width (W)

        Returns
        -------
        Tensor
            [N, H, W, 3], grid of 3D rays
        """
        raise NotImplementedError()


# -------------- Monitor Types --------------


class Plane(Monitor):
    """Plane Monitor"""

    def __init__(
        self,
        init_center_x=0,
        init_center_y=0,
        init_center_z=0.5,
        init_center_std=0.05,
        init_angle_x=0,
        init_angle_y=0,
        init_angle_z=0,
        init_angle_std=0.05,
        eps=1e-5,
    ):
        """
        Parameters
        ----------
        init_center_x : float
            initial center x
        init_center_y : float
            initial center y
        init_center_z : float
            initial center z
        init_center_std : float
            initial center sampling stddev
        init_angle_x : float
            initial angle x
        init_angle_y : float
            initial angle y
        init_angle_z : float
            initial angle z
        init_angle_std : float
            initial angle sampling stddev
        eps : float
            small value used to clip denominator for numerical stability
        """
        super().__init__()

        self.init_center = [
            float(init_center_x),
            float(init_center_y),
            float(init_center_z),
        ]
        self.init_angle = [
            float(init_angle_x),
            float(init_angle_y),
            float(init_angle_z),
        ]
        self.init_center_std = float(init_center_std)
        self.init_angle_std = float(init_angle_std)
        self.eps = float(eps)

        self.center = Parameter(torch.tensor(self.init_center))
        self.angle = Parameter(torch.tensor(self.init_angle))

        self.center_std = Parameter(torch.zeros(3))
        self.angle_std = Parameter(torch.zeros(3))

        for param in [self.center, self.angle, self.center_std, self.angle_std]:
            param.decay = False

        self._restart()

        self._position = dict()
        self._rays = None

    def _restart(self):
        with torch.no_grad():
            self.center_std.fill_(self.init_center_std)
            self.angle_std.fill_(self.init_angle_std)

    def _reset(self):
        self._position.clear()
        self._rays = None

    def position(self, batch_size=1):
        """
        Parameters
        ----------
        batch_size : int
            batch size (N)

        Returns
        -------
        Tensor
            [N, 3], center
        Tensor
            [N, 3], X unit vector
        Tensor
            [N, 3], Y unit vector
        Tensor
            [N, 3], Z unit vector
        """
        if self._position and batch_size == self._position["center"].size(0):
            assert batch_size == self._position["center"].size(0)

        else:
            center = self.center.repeat(batch_size, 1)
            angle = self.angle.repeat(batch_size, 1)

            if self.training:
                center = center + torch.randn_like(center) * self.center_std
                angle = angle + torch.randn_like(angle) * self.angle_std

            X, Y, Z = rmat_3d(*angle.unbind(1)).unbind(2)

            self._position["center"] = center
            self._position["X"] = X
            self._position["Y"] = Y
            self._position["Z"] = Z

        return self._position["center"], self._position["X"], self._position["Y"], self._position["Z"]

    def project(self, rays):
        """
        Parameters
        ----------
        rays : Tensor
            [N, H, W, 3], grid of 3D rays

        Returns
        -------
        Tensor
            [N, H, W, 2], 2D projection
        """
        center, X, Y, Z = self.position(batch_size=rays.size(0))

        a = torch.einsum("N D , N D -> N", Z, center)[:, None, None]
        b = torch.einsum("N D , N H W D -> N H W", Z, rays).clip(self.eps)

        c = torch.einsum("N H W , N H W D -> N H W D", a / b, rays)
        d = c - center[:, None, None, :]

        proj = [
            torch.einsum("N H W D , N D -> N H W", d, X),
            torch.einsum("N H W D , N D -> N H W", d, Y),
        ]
        return torch.stack(proj, dim=3)

    def rays(self, batch_size=1, height=144, width=256):
        """
        Parameters
        ----------
        batch_size : int
            batch size (N)
        height : int
            output height (H)
        width : int
            output width (W)

        Returns
        -------
        Tensor
            [N, H, W, 3], grid of 3D rays
        """
        if self._rays is None:
            x, y = isotropic_grid_2d(
                height=height,
                width=width,
                device=self.device,
            ).unbind(2)

            center, X, Y, _ = self.position(batch_size=batch_size)

            X = torch.einsum("H W , N D -> N H W D", x, X)
            Y = torch.einsum("H W , N D -> N H W D", y, Y)

            self._rays = center[:, None, None, :] + X + Y

        else:
            N, H, W, _ = self._rays.shape
            assert N == batch_size
            assert H == height
            assert W == width

        return self._rays

    def extra_repr(self):
        params = self.center.tolist() + self.angle.tolist()
        return "center=[{:.3g}, {:.3g}, {:.3g}], angle=[{:.3g}, {:.3g}, {:.3g}]".format(*params)
