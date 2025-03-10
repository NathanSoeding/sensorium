import torch
from .modules import Module
from .utils import isotropic_grid_2d


# -------------- Retina Base --------------


class Retina(Module):
    """Retina Module"""

    def _init(self, height=144, width=256):
        """
        Parameters
        ----------
        height : int
            output height (H)
        width : int
            output width (W)
        """
        self.register_buffer("grid", self._grid(height=height, width=width))

    def _grid(self, height=144, width=256):
        """
        Parameters
        ----------
        height : int
            output height (H)
        width : int
            output width (W)

        Returns
        -------
        Tensor
            [H, W, 3], grid of 3D rays
        """
        raise NotImplementedError()

    def project(self, rays, rmat):
        """
        Parameters
        ----------
        rays : Tensor
            [N, H, W, 3], grid of 3D rays
        rmat : Tensor
            [N, 3, 3], 3D rotation matrix

        Returns
        -------
        Tensor
            [N, H, W, 2], 2D projection
        """
        raise NotImplementedError()

    def rays(self, rmat):
        """
        Parameters
        ----------
        rmat : Tensor
            [N, 3, 3], 3D rotation matrix

        Returns
        -------
        Tensor
            [N, H, W, 3], grid of 3D rays
        """
        return torch.einsum("N C D , H W D -> N H W C", rmat, self.grid)


# -------------- Retina Types --------------


class Angular(Retina):
    """Angular Retina"""

    def __init__(self, degrees=75):
        """
        Parameters
        ----------
        degrees : float
            maximum visual degrees
        """
        super().__init__()
        self.degrees = float(degrees)
        self.radians = self.degrees / 180 * torch.pi

    def _grid(self, height=144, width=256):
        """
        Parameters
        ----------
        height : int
            output height (H)
        width : int
            output width (W)

        Returns
        -------
        Tensor
            [H, W, 3], grid of 3D rays
        """
        x, y = isotropic_grid_2d(
            height=height,
            width=width,
            device=self.device,
        ).unbind(2)

        r = torch.sqrt(x.pow(2) + y.pow(2)).mul(self.radians).clip(0, torch.pi / 2)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)

        theta = torch.atan2(y, x)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        grid = [
            sin_r * cos_theta,
            sin_r * sin_theta,
            cos_r,
        ]
        return torch.stack(grid, dim=2)

    def project(self, rays, rmat):
        """
        Parameters
        ----------
        rays : Tensor
            [N, H, W, 3], grid of 3D rays
        rmat : Tensor
            [N, 3, 3], 3D rotation matrix

        Returns
        -------
        Tensor
            [N, H, W, 2], 2D projection
        """
        norm = rays.norm(p=2, dim=3, keepdim=True)
        rays = rays / norm

        x, y, z = torch.einsum("N H W C , N C D -> N H W D", rays, rmat).unbind(3)

        r = torch.arccos(z).div(self.radians)
        theta = torch.atan2(y, x)

        proj = [
            r * torch.cos(theta),
            r * torch.sin(theta),
        ]
        return torch.stack(proj, dim=3)

    def extra_repr(self):
        return f"degrees={self.degrees:.3g}"
