import numpy as np
from iminuit import Minuit, describe
from iminuit.util import make_func_code


class NLL(object):
    """Negative log likelihood for use with Minuit.

    This class pretends that it is a function to Minuit. Note that the error
    definition is a float value of 1/2, giving errors of one standard
    deviation.

    Attributes:
        model: The function that we will minimise the nll of.
        x: The array-like data.
    """
    errordef = Minuit.LIKELIHOOD

    def __init__(self, x, model):
        self.x = x
        self.model = model

        # fake the function signature
        self.func_code = make_func_code(describe(self.model)[1:])

    def __call__(self, *args):
        """Return the nll.

        Arguments:
            args: A variable length list of arguments.

        """
        return -np.sum(np.log(self.model(self.x, *args)))
