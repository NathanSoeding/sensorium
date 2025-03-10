import torch
from .modules import Module
from .parameters import Parameter, ParameterList
from .elements import Conv, InterGroup, Accumulate, Dropout
from .utils import cat_groups_2d


# -------------- Recurrent Base --------------


class Recurrent(Module):
    """Recurrent Module"""

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams, S
        """
        raise NotImplementedError()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream (O)
        """
        raise NotImplementedError()

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        raise NotImplementedError()


# -------------- Recurrent Types --------------


class Rvt(Recurrent):
    """Recurrent Vision Transformer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        common_channels,
        groups=1,
        spatial=3,
        init_gate=1,
        dropout=0,
    ):
        """
        Parameters
        ----------
        in_channels : int
            in channels per stream
        out_channels : int
            out channels per stream
        hidden_channels : int
            hidden channels per stream
        common_channels : int
            common channels per stream
        groups : int
            groups per stream
        spatial : int
            spatial kernel size
        init_gate : float
            initial gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        if in_channels % groups != 0:
            raise ValueError("Input channels must be divisible by groups")

        if hidden_channels % groups != 0:
            raise ValueError("Hidden channels must be divisible by groups")

        if common_channels % groups != 0:
            raise ValueError("Common channels must be divisible by groups")

        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.common_channels = int(common_channels)
        self.group_channels = int(common_channels // groups)

        self.groups = int(groups)
        self.spatial = int(spatial)
        self.init_gate = float(init_gate)
        self._dropout = float(dropout)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams, S
        """
        self._inputs = list(map(int, inputs))
        self.streams = int(streams)

        self.drop_x = Dropout(p=self._dropout)
        self.drop_h = Dropout(p=self._dropout)

        self.conv = Conv(
            in_channels=self.in_channels + self.hidden_channels,
            out_channels=self.common_channels,
            in_groups=self.groups,
            out_groups=self.groups,
            streams=self.streams,
            spatial=self.spatial,
            gain=None,
            bias=None,
        )

        scale = lambda: Parameter(torch.ones([self.groups, self.group_channels]))
        self.scales = ParameterList([scale() for _ in range(self.streams)])
        self.scales.decay = False
        self.scales.norm_dim = 1

        if self.groups > 1:
            gain = (len(self._inputs) + 1) ** -0.5
            intergroup = InterGroup(
                in_channels=self.hidden_channels,
                out_channels=self.in_channels,
                groups=self.groups,
                streams=self.streams,
                gain=gain,
            )
            inputs = [intergroup]
        else:
            gain = len(self._inputs) ** -0.5
            inputs = []

        for i, in_channels in enumerate(self._inputs):
            conv = Conv(
                in_channels=in_channels,
                out_channels=self.in_channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain,
                bias=None if i else 0,
            )
            inputs.append(conv)

        self.proj_x = Accumulate(inputs)

        def proj(in_channels, out_channels, gain, bias):
            return Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain,
                bias=bias,
            )

        self.proj_q = proj(
            in_channels=self.common_channels,
            out_channels=self.common_channels,
            gain=None,
            bias=None,
        )

        self.proj_k = proj(
            in_channels=self.common_channels,
            out_channels=self.common_channels,
            gain=None,
            bias=None,
        )

        self.proj_v = proj(
            in_channels=self.common_channels,
            out_channels=self.common_channels,
            gain=None,
            bias=None,
        )

        self.proj_z = proj(
            in_channels=self.common_channels * 2,
            out_channels=self.hidden_channels,
            gain=1,
            bias=self.init_gate,
        )

        self.proj_n = proj(
            in_channels=self.common_channels * 2,
            out_channels=self.hidden_channels,
            gain=1,
            bias=0,
        )

        self.out = Conv(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            streams=self.streams,
        )

        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream (O)
        """
        return self.out_channels

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        if stream is None:
            S = self.streams
            s = torch.stack(list(self.scales))[:, :, :, None]
        else:
            S = 1
            s = self.scales[stream][None, :, :, None]

        if self.past:
            h = self.past["h"]
        else:
            h = torch.zeros(1, S * self.hidden_channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = torch.tanh(self.proj_x([h, *x], stream=stream))
        else:
            x = torch.tanh(self.proj_x(x, stream=stream))

        xh = cat_groups_2d([self.drop_x(x), h], groups=S * self.groups, expand=True)
        c = self.conv(xh, stream=stream)

        N, _, H, W = c.shape
        HW = H * W

        q = self.proj_q(c, stream=stream).view(N, S, self.groups, self.group_channels, HW)
        k = self.proj_k(c, stream=stream).view(N, S, self.groups, self.group_channels, HW)
        v = self.proj_v(c, stream=stream).view(N, S, self.groups, self.group_channels, HW)

        q = q / q.norm(p=2, dim=3, keepdim=True) * s
        k = k / k.norm(p=2, dim=3, keepdim=True)

        w = torch.einsum("N S G C Q , N S G C D -> N S G Q D", q, k).softmax(dim=-1)
        a = torch.einsum("N S G C D , N S G Q D -> N S G C Q", v, w).view(N, -1, H, W)

        ca = cat_groups_2d([c, a], groups=S * self.groups, expand=True)
        z = torch.sigmoid(self.proj_z(ca, stream=stream))
        n = torch.tanh(self.proj_n(ca, stream=stream))

        h = z * h + (1 - z) * self.drop_h(n)
        self.past["h"] = h

        return self.out(h, stream=stream)


class CvtLstm(Recurrent):
    """Convolutional Vision Transformer Lstm"""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        common_channels,
        groups=1,
        spatial=3,
        init_input=-1,
        init_forget=1,
        dropout=0,
    ):
        """
        Parameters
        ----------
        in_channels : int
            in channels per stream
        out_channels : int
            out channels per stream
        hidden_channels : int
            hidden channels per stream
        common_channels : int
            common channels per stream
        groups : int
            groups per stream
        spatial : int
            spatial kernel size
        init_input : float
            initial input gate bias
        init_forget : float
            initial forget gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        if in_channels % groups != 0:
            raise ValueError("Input channels must be divisible by groups")

        if hidden_channels % groups != 0:
            raise ValueError("Hidden channels must be divisible by groups")

        if common_channels % groups != 0:
            raise ValueError("Common channels must be divisible by groups")

        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.common_channels = int(common_channels)
        self.group_channels = int(common_channels // groups)

        self.groups = int(groups)
        self.spatial = int(spatial)
        self.init_input = float(init_input)
        self.init_forget = float(init_forget)
        self._dropout = float(dropout)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams, S
        """
        self._inputs = list(map(int, inputs))
        self.streams = int(streams)

        self.drop_x = Dropout(p=self._dropout)
        self.drop_h = Dropout(p=self._dropout)

        self.conv = Conv(
            in_channels=self.in_channels + self.hidden_channels,
            out_channels=self.common_channels,
            in_groups=self.groups,
            out_groups=self.groups,
            streams=self.streams,
            spatial=self.spatial,
            gain=None,
            bias=None,
        )

        scale = lambda: Parameter(torch.ones([self.groups, self.group_channels]))
        self.scales = ParameterList([scale() for _ in range(self.streams)])
        self.scales.decay = False
        self.scales.norm_dim = 1

        if self.groups > 1:
            gain = (len(self._inputs) + 1) ** -0.5
            intergroup = InterGroup(
                in_channels=self.hidden_channels,
                out_channels=self.in_channels,
                groups=self.groups,
                streams=self.streams,
                gain=gain,
            )
            inputs = [intergroup]
        else:
            gain = len(self._inputs) ** -0.5
            inputs = []

        for i, in_channels in enumerate(self._inputs):
            conv = Conv(
                in_channels=in_channels,
                out_channels=self.in_channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain,
                bias=None if i else 0,
            )
            inputs.append(conv)

        self.proj_x = Accumulate(inputs)

        def proj(in_channels, out_channels, gain, bias):
            return Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain,
                bias=bias,
            )

        self.proj_q = proj(
            in_channels=self.common_channels,
            out_channels=self.common_channels,
            gain=None,
            bias=None,
        )

        self.proj_k = proj(
            in_channels=self.common_channels,
            out_channels=self.common_channels,
            gain=None,
            bias=None,
        )

        self.proj_v = proj(
            in_channels=self.common_channels,
            out_channels=self.common_channels,
            gain=None,
            bias=None,
        )

        self.proj_i = proj(
            in_channels=self.common_channels * 2,
            out_channels=self.hidden_channels,
            gain=1,
            bias=self.init_input,
        )

        self.proj_f = proj(
            in_channels=self.common_channels * 2,
            out_channels=self.hidden_channels,
            gain=1,
            bias=self.init_forget,
        )

        self.proj_g = proj(
            in_channels=self.common_channels * 2,
            out_channels=self.hidden_channels,
            gain=1,
            bias=0,
        )

        self.proj_o = proj(
            in_channels=self.common_channels * 2,
            out_channels=self.hidden_channels,
            gain=1,
            bias=0,
        )

        self.out = Conv(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            streams=self.streams,
        )

        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream (O)
        """
        return self.out_channels

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        if stream is None:
            S = self.streams
            s = torch.stack(list(self.scales))[:, :, :, None]
        else:
            S = 1
            s = self.scales[stream][None, :, :, None]

        if self.past:
            c = self.past["c"]
            h = self.past["h"]
        else:
            h = c = torch.zeros(1, S * self.hidden_channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = torch.tanh(self.proj_x([h, *x], stream=stream))
        else:
            x = torch.tanh(self.proj_x(x, stream=stream))

        xh = cat_groups_2d([self.drop_x(x), h], groups=S * self.groups, expand=True)
        z = self.conv(xh, stream=stream)

        N, _, H, W = z.shape
        HW = H * W

        q = self.proj_q(z, stream=stream).view(N, S, self.groups, self.group_channels, HW)
        k = self.proj_k(z, stream=stream).view(N, S, self.groups, self.group_channels, HW)
        v = self.proj_v(z, stream=stream).view(N, S, self.groups, self.group_channels, HW)

        q = q / q.norm(p=2, dim=3, keepdim=True) * s
        k = k / k.norm(p=2, dim=3, keepdim=True)

        w = torch.einsum("N S G C Q , N S G C D -> N S G Q D", q, k).softmax(dim=-1)
        a = torch.einsum("N S G C D , N S G Q D -> N S G C Q", v, w).view(N, -1, H, W)

        za = cat_groups_2d([z, a], groups=S * self.groups, expand=True)
        i = torch.sigmoid(self.proj_i(za, stream=stream))
        f = torch.sigmoid(self.proj_f(za, stream=stream))
        g = torch.tanh(self.proj_g(za, stream=stream))
        o = torch.sigmoid(self.proj_o(za, stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop_h(h)

        self.past["c"] = c
        self.past["h"] = h

        return self.out(h, stream=stream)


class ConvLstm(Recurrent):
    """Convolutional Lstm"""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        groups=1,
        spatial=3,
        init_input=-1,
        init_forget=1,
        dropout=0,
    ):
        """
        Parameters
        ----------
        in_channels : int
            in channels per stream
        out_channels : int
            out channels per stream
        hidden_channels : int
            hidden channels per stream
        groups : int
            groups per stream
        spatial : int
            spatial kernel size
        init_input : float
            initial input gate bias
        init_forget : float
            initial forget gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        if in_channels % groups != 0:
            raise ValueError("Input channels must be divisible by groups")

        if hidden_channels % groups != 0:
            raise ValueError("Hidden channels must be divisible by groups")

        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)

        self.groups = int(groups)
        self.spatial = int(spatial)
        self.init_input = float(init_input)
        self.init_forget = float(init_forget)
        self._dropout = float(dropout)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams, S
        """
        self._inputs = list(map(int, inputs))
        self.streams = int(streams)

        self.drop_x = Dropout(p=self._dropout)
        self.drop_h = Dropout(p=self._dropout)

        if self.groups > 1:
            gain = (len(self._inputs) + 1) ** -0.5
            intergroup = InterGroup(
                in_channels=self.hidden_channels,
                out_channels=self.in_channels,
                groups=self.groups,
                streams=self.streams,
                gain=gain,
            )
            inputs = [intergroup]
        else:
            gain = len(self._inputs) ** -0.5
            inputs = []

        for i, in_channels in enumerate(self._inputs):
            conv = Conv(
                in_channels=in_channels,
                out_channels=self.in_channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain,
                bias=None if i else 0,
            )
            inputs.append(conv)

        self.proj_x = Accumulate(inputs)

        def conv(bias):
            return Conv(
                in_channels=self.in_channels + self.hidden_channels,
                out_channels=self.hidden_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                spatial=self.spatial,
                bias=bias,
            )

        self.proj_i = conv(bias=self.init_input)
        self.proj_f = conv(bias=self.init_forget)
        self.proj_g = conv(bias=0)
        self.proj_o = conv(bias=0)

        self.out = Conv(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            streams=self.streams,
        )

        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream (O)
        """
        return self.out_channels

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        if stream is None:
            S = self.streams
        else:
            S = 1

        if self.past:
            c = self.past["c"]
            h = self.past["h"]
        else:
            h = c = torch.zeros(1, S * self.hidden_channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = torch.tanh(self.proj_x([h, *x], stream=stream))
        else:
            x = torch.tanh(self.proj_x(x, stream=stream))

        xh = cat_groups_2d([self.drop_x(x), h], groups=S * self.groups, expand=True)
        i = torch.sigmoid(self.proj_i(xh, stream=stream))
        f = torch.sigmoid(self.proj_f(xh, stream=stream))
        g = torch.tanh(self.proj_g(xh, stream=stream))
        o = torch.sigmoid(self.proj_o(xh, stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop_h(h)

        self.past["c"] = c
        self.past["h"] = h

        return self.out(h, stream=stream)
