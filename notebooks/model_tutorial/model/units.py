import torch
from torch.nn import ELU
from .modules import Module


# -------------- Unit Base --------------


class Unit(Module):
    """Unit Module"""

    @property
    def readouts(self):
        """
        Returns
        -------
        int
            readouts per unit (R)
        """
        raise NotImplementedError()

    def forward(self, readout):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]

        Returns
        -------
        Tensor
            [N, U]
        """
        raise NotImplementedError()

    def loss(self, readout, unit):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]
        unit : Tensor
            [N, U]

        Returns
        -------
        Tensor
            [N, U]
        """
        raise NotImplementedError()


# -------------- Unit Types --------------


class Poisson(Unit):
    """Poisson Unit"""

    @property
    def readouts(self):
        """
        Returns
        -------
        int
            readouts per unit (R)
        """
        return 1

    def forward(self, readout):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]

        Returns
        -------
        Tensor
            [N, U]
        """
        return readout.squeeze(2).exp()

    def loss(self, readout, unit):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]
        unit : Tensor
            [N, U]

        Returns
        -------
        Tensor
            [N, U]
        """
        r = readout.squeeze(2)
        return r.exp() - r * unit


class EluMse(Unit):
    """Elu Mse Unit"""

    def __init__(self, alpha=1):
        """
        Parameters
        ----------
        alpha : float
            alpha value for elu
        """
        super().__init__()
        self.alpha = float(alpha)
        self.elu = ELU(alpha=self.alpha)

    @property
    def readouts(self):
        """
        Returns
        -------
        int
            readouts per unit (R)
        """
        return 1

    def forward(self, readout):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]

        Returns
        -------
        Tensor
            [N, U]
        """
        return self.elu(readout).squeeze(2) + self.alpha

    def loss(self, readout, unit):
        """
        Parameters
        ----------
        readout : Tensor
            [N, U, R]
        unit : Tensor
            [N, U]

        Returns
        -------
        Tensor
            [N, U]
        """
        r = self.elu(readout).squeeze(2) + self.alpha
        return (r - unit).pow(2)
