from torch import nn
from json import dumps, loads


class Parameter(nn.Parameter):
    """Parameter"""

    @property
    def scale(self):
        return getattr(self, "_scale", 1.0)

    @scale.setter
    def scale(self, scale):
        self._scale = float(scale)

    @property
    def decay(self):
        return getattr(self, "_decay", True)

    @decay.setter
    def decay(self, decay):
        self._decay = bool(decay)

    @property
    def norm_dim(self):
        return getattr(self, "_norm_dim", None)

    @norm_dim.setter
    def norm_dim(self, norm_dim):
        if norm_dim is None:
            self._norm_dim = None
        else:
            try:
                self._norm_dim = sorted(map(int, norm_dim))
            except TypeError:
                self._norm_dim = int(norm_dim)

    def __repr__(self):
        return "Parameter containing:\n" + repr(self.data)


class ParameterList(nn.ParameterList):
    """Parameter List"""

    @property
    def scale(self):
        (scale,) = {p.scale for p in self.parameters()}
        return scale

    @scale.setter
    def scale(self, scale):
        for p in self.parameters():
            p.scale = scale

    @property
    def decay(self):
        (decay,) = {p.decay for p in self.parameters()}
        return decay

    @decay.setter
    def decay(self, decay):
        for p in self.parameters():
            p.decay = decay

    @property
    def norm_dim(self):
        (norm_dim,) = {dumps(p.norm_dim) for p in self.parameters()}
        return loads(norm_dim)

    @norm_dim.setter
    def norm_dim(self, norm_dim):
        for p in self.parameters():
            p.norm_dim = norm_dim
