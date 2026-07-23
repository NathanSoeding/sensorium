from neuralpredictors.layers.readouts import (
    MultiReadoutSharedParametersBase,
    FullGaussian2d,
    Factorized2d,
    GeneralizedFullGaussianReadout2d,
)


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d

class MultipleFactorized2d(MultiReadoutSharedParametersBase):
    _base_readout = Factorized2d

class MultipleGeneralizedFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = GeneralizedFullGaussianReadout2d
