import math
import torch
from torch import nn
from functools import reduce


def add(tensors):
    """Adds a list of tensors together

    Parameters
    ----------
    tensor : List[Tensor]
        list of tensors to add together

    Returns
    -------
    Tensor
        tensors added together
    """
    return reduce(torch.Tensor.add, tensors)


def to_groups(tensor, groups):
    """Reshapes a (N, F) tensor to feature groups

    Parameters
    ----------
    tensor : Tensor
        [N, F]
    groups : int
        channel groups (G)

    Returns
    -------
    Tensor
        [N, G, F//G]
    """
    N, _ = tensor.shape
    return tensor.view(N, groups, -1)


def cat_groups(tensors, groups, expand=False):
    """Groupwise concatenation of (N, F) tensors along the feature dimension (F)

    Parameters
    ----------
    tensors : Sequence[Tensor]
        [[N, F], ...]
    groups : int
        feature groups (G)
    expand : bool
        expand batch (N) dimension

    Returns
    -------
    Tensor
        [N, F']
    """
    if expand:
        N, _ = tensors[0].shape
        tensors = tensors[:1] + [t.expand(N, -1) for t in tensors[1:]]

    if groups == 1:
        return torch.cat(tensors, 1)
    else:
        tensors = [to_groups(_, groups) for _ in tensors]
        return torch.cat(tensors, 2).flatten(1, 2)


def to_groups_2d(tensor, groups):
    """Reshapes a 2D (N, C, H, W) tensor to channel groups

    Parameters
    ----------
    tensor : Tensor
        [N, C, H, W]
    groups : int
        channel groups (G)

    Returns
    -------
    Tensor
        [N, G, C//G, H, W]
    """
    N, _, H, W = tensor.shape
    return tensor.view(N, groups, -1, H, W)


def cat_groups_2d(tensors, groups, expand=False):
    """Groupwise concatenation of 2D (N, C, H, W) tensors along the channel dimension (C)

    Parameters
    ----------
    tensors : Sequence[Tensor]
        [[N, C, H, W], ...]
    groups : int
        channel groups (G)
    expand : bool
        expand batch and spatial dimensions

    Returns
    -------
    Tensor
        [N, C', H, W]
    """
    if expand:
        N, _, H, W = tensors[0].shape
        tensors = tensors[:1] + [t.expand(N, -1, H, W) for t in tensors[1:]]

    if groups == 1:
        return torch.cat(tensors, 1)
    else:
        tensors = [to_groups_2d(_, groups) for _ in tensors]
        return torch.cat(tensors, 2).flatten(1, 2)


def rmat_3d(x, y, z):
    """Creates a 3D rotation matrix

    Parameters
    ----------
    x : Tensor
        [N], angle (radians) of rotation around x axis
    y : Tensor
        [N], angle (radians) of rotation around y axis
    z : Tensor
        [N], angle (radians) of rotation around z axis

    Returns
    -------
    Tensor
        [N, 3, 3]
    """
    N = len(x)
    assert N == len(y) == len(z)

    A = torch.eye(3, device=x.device).repeat(N, 1, 1)
    B = torch.eye(3, device=x.device).repeat(N, 1, 1)
    C = torch.eye(3, device=x.device).repeat(N, 1, 1)

    cos_z = torch.cos(z)
    sin_z = torch.sin(z)

    A[:, 0, 0] = cos_z
    A[:, 0, 1] = -sin_z
    A[:, 1, 0] = sin_z
    A[:, 1, 1] = cos_z

    cos_y = torch.cos(y)
    sin_y = torch.sin(y)

    B[:, 0, 0] = cos_y
    B[:, 0, 2] = sin_y
    B[:, 2, 0] = -sin_y
    B[:, 2, 2] = cos_y

    cos_x = torch.cos(x)
    sin_x = torch.sin(x)

    C[:, 1, 1] = cos_x
    C[:, 1, 2] = -sin_x
    C[:, 2, 1] = sin_x
    C[:, 2, 2] = cos_x

    return A.matmul(B).matmul(C)


def isotropic_grid_2d(height, width, major="x", dtype=None, device=None):
    """Creates an isotropic 2d grid

    Parameters
    ----------
    height : int
        grid height (H)
    width : int
        grid width (W)
    major : str
        "x" | "y" , axis by which grid is scaled
    dtype : torch.dtype | None
        grid dtype
    device : torch.device | None
        grid device

    Returns
    -------
    Tensor
        [H, W, 2]
    """
    grid_x = torch.linspace(-1, 1, width, dtype=dtype, device=device)
    grid_y = torch.linspace(-1, 1, height, dtype=dtype, device=device)

    if major == "x":
        grid_y = grid_y * height / width
        scale = (width - 1) / width

    elif major == "y":
        grid_x = grid_x * width / height
        scale = (height - 1) / height

    else:
        raise ValueError("major must be either 'x' or 'y'")

    grid = torch.meshgrid(
        grid_x * scale,
        grid_y * scale,
        indexing="xy",
    )
    return torch.stack(grid, dim=2)


def isotropic_grid_sample_2d(x, grid, major="x", pad_mode="zeros", pad_value=0):
    """Isotropic 2D sampling

    Parameters
    ----------
    x : Tensor
        [N, C, H, W]
    grid : Tensor
        [N, H', W', 2]
    major : str
        "x" | "y" , axis by which sampling is scaled
    pad_mode : str
        "zeros" | "replicate" | "constant" -- padding mode for out-of-bounds grid values
    pad_value : float
        value of padding when pad_mode == "constant"

    Returns
    -------
    Tensor
        [N, C, H', W']
    """
    if pad_mode != "constant" and pad_value:
        raise ValueError("can only specify pad_value when pad_mode == 'constant'")

    if pad_mode == "zeros":
        finalize = lambda x: x
        padding_mode = "zeros"

    elif pad_mode == "replicate":
        finalize = lambda x: x
        padding_mode = "border"

    elif pad_mode == "constant":
        x = x - pad_value
        finalize = lambda x: x + pad_value
        padding_mode = "zeros"

    else:
        raise ValueError("pad_mode must either be 'zeros', 'replicate', or 'constant'")

    _, _, height, width = x.shape
    grid_x, grid_y = grid.unbind(dim=3)

    if major == "x":
        grid_y = grid_y * width / height
        scale = width / (width - 1)

    elif major == "y":
        grid_x = grid_x * height / width
        scale = height / (height - 1)

    else:
        raise ValueError("major must be either 'x' or 'y'")

    _, height, width, _ = grid.shape
    grid = [
        grid_x * scale * (width - 1) / width,
        grid_y * scale * (height - 1) / height,
    ]

    x = nn.functional.grid_sample(
        x,
        grid=torch.stack(grid, dim=3),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )
    return finalize(x)


class Gaussian3d(nn.Module):
    """3D (Spatiotemporal) Gaussian Blur"""

    def __init__(self, spatial_std=1, temporal_std=1, cutoff=4):
        """
        Parameters
        ----------
        spatial_sigma : float
            spatial standard deviation
        temporal_sigma : float
            temporal standard deviation
        cutoff : float
            standard deviation cutoff
        """
        from scipy.signal.windows import gaussian

        super().__init__()

        self.spatial_std = float(spatial_std)
        self.temporal_std = float(temporal_std)
        self.cutoff = float(cutoff)

        def pad_kernel(std):
            M = math.ceil(std * cutoff) * 2 + 1
            pad = M // 2
            kernel = gaussian(M=M, std=std)
            kernel = torch.tensor(kernel / kernel.sum(), dtype=torch.float)
            return pad, kernel

        self.spatial_pad, spatial_kernel = pad_kernel(self.spatial_std)
        self.temporal_pad, temporal_kernel = pad_kernel(self.temporal_std)

        self.pads = [
            [0, 0, 0, 0, self.temporal_pad, self.temporal_pad],
            [0, 0, self.spatial_pad, self.spatial_pad, 0, 0],
            [self.spatial_pad, self.spatial_pad, 0, 0, 0, 0],
        ]

        self.register_buffer("spatial_kernel", spatial_kernel)
        self.register_buffer("temporal_kernel", temporal_kernel)

    @property
    def kernels(self):
        return [
            self.temporal_kernel[None, None, :, None, None],
            self.spatial_kernel[None, None, None, :, None],
            self.spatial_kernel[None, None, None, None, :],
        ]

    def forward(self, x):
        """
        Parameters
        ----------
        x : 4D Tensor
            [N, T, H, W]

        Returns
        -------
        4D Tensor
            [N, T, H, W]
        """
        channels, _, _, _ = x.shape
        x = x.unsqueeze(dim=0)

        for p, k in zip(self.pads, self.kernels):

            x = nn.functional.conv3d(
                input=nn.functional.pad(x, pad=p, mode="replicate"),
                weight=k.expand(channels, -1, -1, -1, -1),
                groups=channels,
            )

        return x.squeeze(dim=0)

    def extra_repr(self):
        return f"spatial_std={self.spatial_std:.3g}, temporal_std={self.temporal_std:.3g}"
