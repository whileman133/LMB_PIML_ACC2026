"""
lumped.base.py

Base definitions for lumped-parameter models.
"""

import pybamm


class _Container:
    """
    Generic container class for parameters and variables.
    """

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            raise AttributeError(f"Cannot find {item} in model.")

    def __setattr__(self, key, value):
        return super().__setattr__(key, value)


class BaseLumpedModel(pybamm.BaseModel):
    """
    Base class for lumped-parameter models.
    """

    def __init__(self, name="Unnamed lumped-parameter model"):
        super().__init__(name)
        self.param = _Container()    # model parameters
        self.var = _Container()      # model variables