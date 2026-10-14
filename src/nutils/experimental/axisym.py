from nutils import function, evaluable, transformseq
import numpy as np
from functools import partial

try:
    from nutils.units.core import Monomial
except ImportError:
    Monomial = None


class Lower(function.Array):
    """Lower an Array using predefined lowering arguments."""

    def __init__(self, func, lower_args):
        assert isinstance(lower_args, function.LowerArgs)
        assert isinstance(func, function.Array)
        self._func = func
        self._lower_args = lower_args
        super().__init__(
            shape=func.shape,
            dtype=func.dtype,
            arguments=func.arguments,
            spaces=func.spaces - lower_args.spaces,
        )

    def lower(self, args: function.LowerArgs):
        return self._func.lower(args * self._lower_args)


def lower_by(lower_args, func):
    """Create a (nested, dimensional) Lower instance."""

    cls = type(func)
    if cls in (list, tuple):
        return cls(lower_by(lower_args, item) for item in func)
    elif Monomial and issubclass(cls, Monomial):
        func, dim = func.unwrap()
        while cls.__init__ is not Monomial.__init__:
            (cls,) = cls.__bases__
        return cls(lower_by(lower_args, func), dim)
    else:
        return Lower(function.asarray(func), lower_args)


def gen_R(θ, order):
    """Create a 1x2 or 2x3 transformation matrix.

    The resulting transformation matrix can be used to convert a coordinate
    system defined by the order parameter to the corresponding higher
    dimensional cartesian coordinate (right multiplication) or visa versa (left
    multiplication)."""

    assert "r" in order
    er = [np.cos(θ), np.sin(θ)]
    if order == "r":
        # x=rcosθ, y=rsinθ
        R = np.stack(er)[np.newaxis]  # shape: 1x2
    elif order in ("rx", "ry", "rz", "xr", "yr", "zr"):
        # rx/xr: x=x, y=rcosθ, z=rsinθ
        # ry/yr: x=rcosθ, y=y, z=rsinθ
        # zr/rz: x=rcosθ, y=rsinθ, z=z
        ir = order.index("r")
        jo = "xyz".index(order[1 - ir])
        er.insert(jo, 0.0)
        eo = [0.0, 0.0]
        eo.insert(jo, 1.0)
        E = [np.stack(er), np.stack(eo)]
        if ir == 1:
            E.reverse()
        R = np.stack(E)  # shape: 2x3
    else:
        raise NotImplementedError(f"order={order}")
    return R


def gen_axi(order, space="H"):
    """Create the axial symmetry instrument.

    The returned callable removes the topological space from its argument and
    sets the corresponding angle coordinate to zero. Additionally, the .θ
    attribute contains the angle and the .R attribute the transformation matrix
    as defined by the order parameter."""

    transforms = transformseq.IndexTransforms(ndims=1, length=1)
    lower_args = function.LowerArgs.for_space(
        space,
        transforms=(transforms,),
        index=evaluable.constant(0),
        coordinates=evaluable.Zeros((evaluable.constant(1),), dtype=float),
    )
    (θ,) = function.transforms_coords(space, transforms) * (2 * np.pi)
    axi = partial(lower_by, lower_args)
    axi.θ = θ
    axi.R = gen_R(θ, order)
    return axi
