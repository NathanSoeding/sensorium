import torch
from .modules import Module
from .parameters import Parameter


# -------------- Pixel Base --------------


class Pixel(Module):
    """Pixel Intensity"""

    def forward(self, pixels):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        raise NotImplementedError()

    def inverse(self, pixels):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        raise NotImplementedError()


# -------------- Luminance Types --------------


class Raw(Pixel):
    """Raw Pixel Intensity"""

    def forward(self, pixels):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        return pixels

    def inverse(self, pixels):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        return pixels


class StaticPower(Pixel):
    """Static Power Transform"""

    def __init__(self, power=1, scale=1, offset=0):
        """
        Parameters
        ----------
        power : float
            pixel power
        scale : float
            pixel scale
        offset : float
            pixel offset
        """
        super().__init__()
        self.register_buffer("power", torch.tensor(power, dtype=torch.float))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))
        self.register_buffer("offset", torch.tensor(offset, dtype=torch.float))

    def forward(self, pixels):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        return pixels.pow(self.power).mul(self.scale).add(self.offset)

    def inverse(self, pixels):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        return pixels.sub(self.offset).div(self.scale).pow(1 / self.power)

    def extra_repr(self):
        return f"power={self.power:.3g}, scale={self.scale:.3g}, offset={self.offset:.3g}"


class SigmoidPower(Pixel):
    """Learned (Sigmoid) Power Transform"""

    def __init__(self, max_power=1, init_scale=1, init_offset=0, eps=1e-5):
        """
        Parameters
        ----------
        max_power : float
            maximum pixel power
        init_scale : float
            initial pixel scale
        init_offset : float
            initial pixel offset
        eps : float
            small number for numerical stability
        """
        super().__init__()
        self.max_power = float(max_power)
        self.init_scale = float(init_scale)
        self.init_offset = float(init_offset)
        self.eps = float(eps)

        self.logit = Parameter(torch.zeros(1))
        self.scale = Parameter(torch.full([1], self.init_scale))
        self.offset = Parameter(torch.full([1], self.init_offset))

        for param in [self.logit, self.scale, self.offset]:
            param.decay = False

    @property
    def power(self):
        return self.logit.sigmoid() * self.max_power

    def forward(self, pixels):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        return pixels.add(self.eps).pow(self.power).mul(self.scale).add(self.offset)

    def inverse(self, pixels):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        return pixels.sub(self.offset).div(self.scale).pow(1 / self.power).sub(self.eps)

    @torch.no_grad()
    def extra_repr(self):
        return f"power={self.power.item():.3g}, scale={self.scale.item():.3g}, offset={self.offset.item():.3g}, eps={self.eps:.3g}"
