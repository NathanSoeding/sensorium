from .modules import Module


# -------------- Reduce Base --------------


class Reduce(Module):
    """Reduce Module"""

    def _init(self, dim, keepdim=False):
        """
        Parameters
        ----------
        dim : Sequence[int]
            dimensions to reduce
        keepdim : bool
            whether the output tensor has dim retained or not
        """
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


# -------------- Reduce Types --------------


class Mean(Reduce):
    """Mean Reduction"""

    def _init(self, dim, keepdim=False):
        """
        Parameters
        ----------
        dim : Sequence[int]
            dimensions to reduce
        keepdim : bool
            whether the output tensor has dim retained or not
        """
        self.dim = list(map(int, dim))
        self.keepdim = bool(keepdim)

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=self.keepdim)
