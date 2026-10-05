from neuralpredictors.layers.readouts import MultiReadoutSharedParametersBase, FullGaussian2d, FullFactorized2d, RetinotopicFactorizedLinear2d


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d

class MultipleFactorized2d(MultiReadoutSharedParametersBase):
    _base_readout = FullFactorized2d

class MultipleRetinotopicFactorizedLinear2d(MultiReadoutSharedParametersBase):
    _base_readout = RetinotopicFactorizedLinear2d