import os

import sys
import warnings
import pandas as pd
import torch

#from modulusDL.models.arch import CustomModuleArch, FullyConnectedFlexiLayerSizeArch,
#from modulusDL.eq.pde import NavierStokes_CoordTransformed
import shutil
            
import numpy as np
from sympy import (
    Symbol,
    Function,
    Eq,
    Number,
    Abs,
    Max,
    Min,
    sqrt,
    pi,
    sin,
    cos,
    atan,
    atan2,
    acos,
    asin,
    sign,
    exp,
)

import modulus
import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry import Bounds
from modulus.sym.geometry.geometry import Geometry, csg_curve_naming
from modulus.sym.geometry.curve import SympyCurve
from modulus.sym.geometry.helper import _sympy_sdf_to_sdf
from modulus.sym.geometry.parameterization import Parameterization, Parameter, Bounds
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.geometry.primitives_2d import Line, Circle, Channel2D
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer, Scaler
from modulus.sym.eq.pde import PDE
from modulus.sym.eq.pde import PDE
from modulus.sym.models.arch import Arch
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import logging
import ast

from termcolor import colored
from inspect import signature, _empty
from typing import Optional, Callable, List, Dict, Union, Tuple
from modulus.sym.constants import NO_OP_SCALE
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.constants import JIT_PYTORCH_VERSION
from modulus.sym.distributed import DistributedManager
from modulus.sym.manager import JitManager, JitArchMode
from modulus.sym.models.activation import Activation
from modulus.sym.utils.sympy.functions import parabola

class incrorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):#cline,radius_y
        return torch.cat((x[...,0:1]**2.,x[...,1:2]**2.,x[...,0:1]*x[...,1:2],x[...,0:1]*x[...,2:3],x[...,1:2]*x[...,2:3]),-1)





class CustomModuleArch(Arch):
    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=input_keys,
                output_keys=output_keys,
                detach_keys=detach_keys,
                periodicity=None,
            )
        self.module=nn.ModuleList()
        if module is None:
            self.module.append(torch.nn.Identity())
        else:
            self.module.append(module)
        
    def _tensor_forward(self, x: Tensor) -> Tensor:
        x = self.process_input(
            x,
            self.input_scales_tensor,
            periodicity=self.periodicity,
            input_dict=self.input_key_dict,
            dim=-1,
        )
        x = self.module[0](x)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
         

   detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        x = self.module[0](x)
        x = self.process_output(x, self.output_scales_tensor)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )






class NavierStokes_CoordTransformed(PDE):
    """
    This class is adapted from NVIDIAModulus v22.09 pde.NavierStokes
    the coordinates are adapted to allow for user defined str
    Compressible Navier Stokes equations

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the Navier-Stokes equations.

    Examples
    ========
    >>> ns = NavierStokes(nu=0.01, rho=1, dim=2)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
      momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
    >>> ns = NavierStokes(nu='nu', rho=1, dim=2, time=False)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: -nu*u__x__x - nu*u__y__y + u*u__x + v*u__y - nu__x*u__x - nu__y*u__y + p__x
      momentum_y: -nu*v__x__x - nu*v__y__y + u*v__x + v*v__y - nu__x*v__x - nu__y*v__y + p__y
    """

    name = "NavierStokes_CoordTransformed"

    def __init__(self, nu,case_coord_strList=None,case_param_strList=None, rho=1, dim=3, time=True, mixed_form=False):
        # set params
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form
        if case_param_strList is None:
            case_param_strList={}
        if case_coord_strList is None:
            case_coord_strList=["x_case","y_case","z_case"]
        if (case_coord_strList)==1:
            case_coord_strList=case_coord_strList+["y_case","z_case"]
        elif (case_coord_strList)==2:
            case_coord_strList=case_coord_strList+["z_case"]
        # coordinates
        t= Symbol("t")

        x = Symbol(case_coord_strList[0])
        y = Symbol(case_coord_strList[1])
        z = Symbol(case_coord_strList[2])
        input_variables = {case_coord_strList[0]: x, case_coord_strList[1]: y, case_coord_strList[2]: z, "t": t}
        for key in case_param_strList:
            input_variables[key]=case_param_strList[key]
        if self.dim == 2:
            input_variables.pop("z_case")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # dynamic viscosity
        mu = rho * nu

        # set equations
        self.equations = {}
        self.equations["continuity"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        )

        if not self.mixed_form:
            curl = Number(0) if rho.diff(x) == 0 else u.diff(x) + v.diff(y) + w.diff(z)
            self.equations["momentum_x"] = (
                (rho * u).diff(t)
                + (
                    u * ((rho * u).diff(x))
                    + v * ((rho * u).diff(y))
                    + w * ((rho * u).diff(z))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u.diff(x)).diff(x)
                - (mu * u.diff(y)).diff(y)
                - (mu * u.diff(z)).diff(z)
                - (mu * (curl).diff(x))
            )
            self.equations["momentum_y"] = (
                (rho * v).diff(t)
                + (
                    u * ((rho * v).diff(x))
                    + v * ((rho * v).diff(y))
                    + w * ((rho * v).diff(z))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v.diff(x)).diff(x)
                - (mu * v.diff(y)).diff(y)
                - (mu * v.diff(z)).diff(z)
                - (mu * (curl).diff(y))
            )
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w).diff(x))
                    + v * ((rho * w).diff(y))
                    + w * ((rho * w).diff(z))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w.diff(x)).diff(x)
                - (mu * w.diff(y)).diff(y)
                - (mu * w.diff(z)).diff(z)
                - (mu * (curl).diff(z))
            )

            if self.dim == 2:
                self.equations.pop("momentum_z")

        elif self.mixed_form:
            u_x = Function("u_x")(*input_variables)
            u_y = Function("u_y")(*input_variables)
            u_z = Function("u_z")(*input_variables)
            v_x = Function("v_x")(*input_variables)
            v_y = Function("v_y")(*input_variables)
            v_z = Function("v_z")(*input_variables)

            if self.dim == 3:
                w_x = Function("w_x")(*input_variables)
                w_y = Function("w_y")(*input_variables)
                w_z = Function("w_z")(*input_variables)
            else:
                w_x = Number(0)
                w_y = Number(0)
                w_z = Number(0)
                u_z = Number(0)
                v_z = Number(0)

            curl = Number(0) if rho.diff(x) == 0 else u_x + v_y + w_z
            self.equations["momentum_x"] = (
                (rho * u).diff(t)
                + (
                    u * ((rho * u.diff(x)))
                    + v * ((rho * u.diff(y)))
                    + w * ((rho * u.diff(z)))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u_x).diff(x)
                - (mu * u_y).diff(y)
                - (mu * u_z).diff(z)
                - (mu * (curl).diff(x))
            )
            self.equations["momentum_y"] = (
                (rho * v).diff(t)
                + (
                    u * ((rho * v.diff(x)))
                    + v * ((rho * v.diff(y)))
                    + w * ((rho * v.diff(z)))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v_x).diff(x)
                - (mu * v_y).diff(y)
                - (mu * v_z).diff(z)
                - (mu * (curl).diff(y))
            )
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w.diff(x)))
                    + v * ((rho * w.diff(y)))
                    + w * ((rho * w.diff(z)))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w_x).diff(x)
                - (mu * w_y).diff(y)
                - (mu * w_z).diff(z)
                - (mu * (curl).diff(z))
            )
            self.equations["compatibility_u_x"] = u.diff(x) - u_x
            self.equations["compatibility_u_y"] = u.diff(y) - u_y
            self.equations["compatibility_u_z"] = u.diff(z) - u_z
            self.equations["compatibility_v_x"] = v.diff(x) - v_x
            self.equations["compatibility_v_y"] = v.diff(y) - v_y
            self.equations["compatibility_v_z"] = v.diff(z) - v_z
            self.equations["compatibility_w_x"] = w.diff(x) - w_x
            self.equations["compatibility_w_y"] = w.diff(y) - w_y
            self.equations["compatibility_w_z"] = w.diff(z) - w_z
            self.equations["compatibility_u_xy"] = u_x.diff(y) - u_y.diff(x)
            self.equations["compatibility_u_xz"] = u_x.diff(z) - u_z.diff(x)
            self.equations["compatibility_u_yz"] = u_y.diff(z) - u_z.diff(y)
            self.equations["compatibility_v_xy"] = v_x.diff(y) - v_y.diff(x)
            self.equations["compatibility_v_xz"] = v_x.diff(z) - v_z.diff(x)
            self.equations["compatibility_v_yz"] = v_y.diff(z) - v_z.diff(y)
            self.equations["compatibility_w_xy"] = w_x.diff(y) - w_y.diff(x)
            self.equations["compatibility_w_xz"] = w_x.diff(z) - w_z.diff(x)
            self.equations["compatibility_w_yz"] = w_y.diff(z) - w_z.diff(y)

            if self.dim == 2:
                self.equations.pop("momentum_z")
                self.equations.pop("compatibility_u_z")
                self.equations.pop("compatibility_v_z")
                self.equations.pop("compatibility_w_x")
                self.equations.pop("compatibility_w_y")
                self.equations.pop("compatibility_w_z")
                self.equations.pop("compatibility_u_xz")
                self.equations.pop("compatibility_u_yz")
                self.equations.pop("compatibility_v_xz")
                self.equations.pop("compatibility_v_yz")
                self.equations.pop("compatibility_w_xy")
                self.equations.pop("compatibility_w_xz")
                self.equations.pop("compatibility_w_yz")







class HLine(Geometry):
    """
    2D Line parallel to y-axis

    Parameters
    ----------
    point_1 : tuple with 2 ints or floats
        lower bound point of line segment
    point_2 : tuple with 2 ints or floats
        upper bound point of line segment
    normal : int or float
        normal direction of line (+1 or -1)
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, normal=1, parameterization=Parameterization()):
        assert point_1[1] == point_2[1], "Points must have same y-coordinate"

        # make sympy symbols to use
        l = Symbol(csg_curve_naming(0))
        y = Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({l: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_y = point_2[0] - point_1[0]
        line_1 = SympyCurve(
            functions={
                "x": point_1[0] + l * dist_y,
                "y": point_1[1],
                "normal_x": 0,  # TODO rm 1e-10
                "normal_y": 1e-10 + normal,
            },
            parameterization=curve_parameterization,
            area=dist_y,
        )
        curves = [line_1]

        # calculate SDF
        sdf = normal * (point_1[1] - y)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize Line
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )
class Channel2D_centerfocused(Geometry):
    """
    2D Channel (no bounding curves in x-direction)

    Parameters
    ----------
    point_1 : tuple with 2 ints or floats
        lower bound point of channel
    point_2 : tuple with 2 ints or floats
        upper bound point of channel
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols to use
        l = Symbol(csg_curve_naming(0))
        x = Symbol("x")
        y = Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({l: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]
        line_1 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_1[1],
                "normal_x": 0,
                "normal_y": -1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        line_2 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_2[1],
                "normal_x": 0,
                "normal_y": 1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        curves = [line_1, line_2]

        # calculate SDF
        center_y = point_1[1] + (dist_y) / 2
        center_x = point_1[0] + (dist_x) / 2
        y_diff = Abs(y - center_y) - (point_2[1] - center_y)
        outside_distance = sqrt(Max(y_diff, 0) ** 2)
        inside_distance = Min(y_diff, 0)
        sdf = -(outside_distance + inside_distance)*(0.2+0.8*exp(-12.5*((x-center_x)/dist_x)**2.))#sigma=0.02

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize Channel2D
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )


class ParabolicInlet(PDE):
    def __init__(self, r_ref,l_ref,umax):
        # coordinates
        dissq = Symbol("dissq")
        
        # make input variables

        # make u function
        u = Symbol("u")
        

        # source term
        umax=Number(umax)
        
        # set equations
        self.equations = {}
        self.equations["parabolic_inlet"] = (
            u- umax*dissq
        )  # "custom_pde" key name will be used in constraints
        
        
class Stenosis_2(torch.nn.Module):
    def __init__(self, r, l):
        super().__init__()
        self.register_buffer("r", torch.tensor(r), persistent=False)
        self.register_buffer("l", torch.tensor(l), persistent=False)  # length of line
       
    def forward(self, x):
        x_ref = x[..., 0:1]
        y_ref = x[..., 1:2]
        fc = x[..., 2:3]
        
        # Transform x_ref to range [-1, 1] for the cosine function
        x_ref_transformed = 2 * (x_ref - self.l/8) / (3*self.l/8 - self.l/8) - 1


        # Create masks for the piecewise function
        mask1 = (x_ref >= -self.l/2) & (x_ref < -self.l / 8)
        mask2 = (x_ref >= -self.l / 8) & (x_ref <  self.l / 8)

        # Compute y_case based on the interval
        y_case = torch.where(
            mask1,
            y_ref,
            torch.where(
                mask2,
                y_ref * (1 - fc * (1 + torch.cos(x_ref_transformed * torch.pi))),
                y_ref
            )
        )

        return torch.cat((x_ref, y_case), -1)
       
class GE(torch.nn.Module):
    def __init__(self, r, l):
        super().__init__()
        r = torch.tensor(r)
        self.register_buffer("r", r, persistent=False)
        l = torch.tensor(l)
        self.register_buffer("l", l, persistent=False)  # length of line

    def forward(self, x):
        x_case = x[..., 0:1]
        y_case = x[..., 1:2]
        fc = x[..., 2:3]
        
        # Transform x_ref to range [-1, 1] for the cosine function
        x_ref_transformed = 2 * (x_case - self.l/8) / (3*self.l/8 - self.l/8) - 1

        # Create masks for the piecewise function
        mask1 = (x_case >= -self.l/2) & (x_case < -self.l / 8)
        mask2 = (x_case >= -self.l / 8) & (x_case < self.l / 8)
        
        radius = torch.where(
            mask1,
            y_case / self.r,
            torch.where(
                mask2,
                y_case / (1 - fc * (1 + torch.cos(x_ref_transformed * torch.pi))) /self.r,
                y_case / self.r
            )
        )
        
        cline = 2. * x_case / self.l

        

        return torch.cat((cline, radius), -1)     
        
class Dissq(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 1.-x[...,0:1]**2.
        
        
class warped(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x[...,2:4]-x[...,0:2]
        
        
class Newcoord(torch.nn.Module):
    def __init__(self,s):
        super().__init__()
        s=torch.tensor(s)
        self.register_buffer("s", s, persistent=False)
    def forward(self,x):
        return self.s*x[...,0:3]  
        
        
        
@modulus.sym.main(config_path="./", config_name="config.yaml")
def run(cfg: ModulusConfig) -> None:
    fc= Symbol("fc")
    inlet_u= Symbol("inlet_u")
    param_ranges = {
    fc: np.array([[0.1],
                           [0.125],
                           [0.15],
                           [0.175],
                           [0.2],
                           [0.225],
                           [0.25],
                           [0.275],
                           [0.3]
                           ]),

    inlet_u: np.array([[0.38],
                           [0.39],
                           [0.4],
                           [0.41],
                           [0.42],
                           [0.43],
                           [0.44],
                           [0.45],
                           [0.46],
                           [0.47],
                           [0.48],
                           [0.49],
                           [0.5],
                           [0.51],
                           [0.52],
                           [0.53],
                           [0.54],
                           [0.55],
                           [0.56],
                           [0.57],
                           [0.58],
                           [0.59],
                           [0.6],
                           [0.61],
                           [0.62],
                           [0.63],
                           [0.64],
                           [0.65],
                           [0.66],
                           [0.67],
                           [0.68],
                           [0.69],
                           [0.7],
                           [0.71],
                           [0.72],
                           [0.73],
                           [0.74],
                           [0.75],
                           [0.76],
                           [0.77],
                           [0.78],
                           [0.79],
                           [0.8],
                           [0.81],
                           [0.82],
                           [0.83],
                           [0.84],
                           [0.85],
                           [0.86],
                           [0.87],
                           [0.88],
                           [0.89],
                           [0.9],
                           [0.91],
                           [0.92],
                           [0.93],
                           [0.94],
                           [0.95],
                           [0.96],
                           [0.97],
                           [0.98],
                           [0.99],
                           [1.0],
                           [1.01],
                           [1.02],
                           [1.03],
                           [1.04],
                           [1.05],
                           [1.06],
                           [1.07],
                           [1.08],
                           [1.09],
                           [1.1],
                           [1.11],
                            [1.12],
                           [1.13],
                           [1.14],
                           [1.15],
                           [1.16],
                           [1.17],
                           [1.18],
                           [1.19],
                           [1.2],
                           [1.21],
                           [1.22],
                           [1.23],
                           [1.24],
                           [1.25],
                           [1.26],
                           [1.27],
                           [1.28],
                           [1.29],
                           [1.3],
                           [1.31],
                           [1.32],
                           [1.33],
                           
                          
                           ]),
    

}
    
    # physical quantities
    mi = 0.004             #kg/(m*s)
    rho = 1060              #kg/m^3
    nu=mi/rho
    
    
    noslip_u = 0
    noslip_v = 0
    outlet_p = 0
    
   
    channel_radius=0.005
    channel_center=(0,0)
    channel_length = (-4*0.005, 4*0.005)
    channel_width = (-1*0.005, 1*0.005)
    
    #scales
    length_scale=0.01
    velocity_scale=1.52
    time_scale=length_scale/velocity_scale
    density_scale=rho
    kinematic_viscocity_scale= length_scale**2/time_scale
    
    #non_dim quantities
    
    channel_length_nd=(-4*0.005/length_scale, 4*0.005/length_scale)
    channel_width_nd=(-1*0.005/length_scale, 1*0.005/length_scale)
    channel_radius_nd= channel_radius/length_scale
    
    inlet_u_nd=inlet_u/velocity_scale
    
    rho_nd=rho/density_scale
    nu_nd=nu/kinematic_viscocity_scale
    
    
    x, y = Symbol("x"), Symbol("y")
    
    pr = Parameterization(param_ranges)
    
    channel = Channel2D_centerfocused(
        (channel_length_nd[0], channel_width_nd[0]),  #x1,y1
        (channel_length_nd[1], channel_width_nd[1]),  #x2,y2
        parameterization=pr,
    )
    
    inlet_2d = Line(
        (channel_length_nd[0], channel_width_nd[0]),  #x1,y1
        (channel_length_nd[0], channel_width_nd[1]),  #x1,y2
        normal=-1,
        parameterization=pr,
    )
    
    outlet_2d = Line(
        (channel_length_nd[1], channel_width_nd[0]),  #x2,y1
        (channel_length_nd[1], channel_width_nd[1]),  #x2,y2
        normal=1,
        parameterization=pr,
    )
    
    wall_btm = HLine(
        (channel_length_nd[0], channel_width_nd[0]),  #x1, y1
        (channel_length_nd[1], channel_width_nd[0]),  #x2, y1
        normal=-1,
        parameterization=pr,
    )
    
    wall_top = HLine(
        (channel_length_nd[0], channel_width_nd[1]),  #x1,y2
        (channel_length_nd[1], channel_width_nd[1]),  #x2,y2
        normal=1,
        parameterization=pr,
    )
    
    volume_geo = channel
    
    
    # make list of nodes to unroll graph on
    Stenosis_coordTransform=CustomModuleArch(
        [Key("x"), Key("y"),Key("fc")],
        [Key("x_case"), Key("y_case")],
        module=Stenosis_2(channel_radius_nd,channel_length_nd[1]-channel_length_nd[0])
        )
        
    
    ge_net=CustomModuleArch(
        [Key("x_case"), Key("y_case"), Key("fc")],
        [Key("cline"), Key("radius")],
        module=GE(channel_radius_nd,channel_length_nd[1]-channel_length_nd[0])
        )
        
        
    warp=CustomModuleArch(
        [Key('x'),Key('y'),Key("x_case"), Key("y_case")],
        [Key("warpx"), Key("warpy")],
        module=warped()
        )
        
    ns = NavierStokes_CoordTransformed(nu=nu_nd, rho=rho_nd, dim=2, time=False)
    
    normal_dot_vel = NormalDotVec(["u", "v"])
    
    
    Dissq_net=CustomModuleArch(
        [Key("radius")],
        [Key("dissq")],
        module=Dissq()
        )
        
    flow_net = instantiate_arch(
        input_keys=[Key("x_case"), Key("y_case"),Key("cline"), Key("radius"),Key("dissq"),Key("cline_sq"),Key("radius_sq"),Key("cline_radius"),Key("cline_dissq"),Key("radius_dissq"),Key("fc"),Key("inlet_u")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
        
    )
    
    
    incrorder_NN = CustomModuleArch(
        input_keys=[Key("cline"),Key("radius"),Key("dissq")],
        output_keys=[Key("cline_sq"),Key("radius_sq"),Key("cline_radius"),Key("cline_dissq"),Key("radius_dissq")],
        module=incrorder(),
    )
    
    nodes = (
        [Stenosis_coordTransform.make_node(name='coordtransform')]
        +[Dissq_net.make_node(name='Dissq_net')]
        +[ge_net.make_node(name='ge_net')]
        #+[all_hb.make_node(name='all_hb')]
        +[incrorder_NN.make_node(name='incrorder_NN')]
        + ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + [warp.make_node(name="newcoord")]
        
        
        #+inlet_eq.make_nodes()
    )
    
    domain = Domain()
    
    batchsizefactor=1
    
    inlet_parabola = parabola(
        y, inter_1=channel_width_nd[0], inter_2=channel_width_nd[1], height=inlet_u_nd
    )    
    
    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_2d,
        outvar={"u": inlet_parabola, "v": 0},
        batch_size=cfg.batch_size.inlet*batchsizefactor,
        parameterization=param_ranges,
    )
    domain.add_constraint(inlet, "inlet")
    
    
    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_2d,
        outvar={"p": outlet_p},
        batch_size=cfg.batch_size.outlet*batchsizefactor,
        parameterization=param_ranges,
    )
    domain.add_constraint(outlet, "outlet")
    
    
    # no slip
    no_slip_wall_btm = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_btm,
        outvar={"u": noslip_u, "v": noslip_v},
        batch_size=cfg.batch_size.walls*batchsizefactor,
        parameterization=param_ranges,
    )
    domain.add_constraint(no_slip_wall_btm, "no_slip_wall")
    
    
    no_slip_wall_top = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_top,
        outvar={"u": noslip_u, "v": noslip_v},
        batch_size=cfg.batch_size.walls*batchsizefactor,
        parameterization=param_ranges,
    )
    domain.add_constraint(no_slip_wall_top, "no_slip_wall")
    
    
    # interior contraints
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior*batchsizefactor,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        parameterization=param_ranges,
    )
    domain.add_constraint(interior, "interior")
    
    
    
             
    
    
    
    
    
    file_path = "/data/Re500/fc0.1/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST1")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    file_path = "/data/Re500/fc0.2/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.2)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST2")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    file_path = "/data/Re500/fc0.3/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.3)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST3")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
        
        
    
    
    
    
    
    
    
    file_path = "/data/Re750/fc0.1/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 1.5*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST4")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    file_path = "/data/Re750/fc0.2/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.2)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 1.5*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST5")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    file_path = "/data/Re750/fc0.3/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.3)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 1.5*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST6")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    file_path = "/data/Re1000/fc0.1/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 2*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST7")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    file_path = "/data/Re1000/fc0.2/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.2)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 2*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST8")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    file_path = "/data/Re1000/fc0.3/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.3)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 2*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST9")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    file_path = "/data/Re1250/fc0.1/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 2.5*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST10")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    file_path = "/data/Re1250/fc0.2/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.2)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 2.5*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST11")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    file_path = "/data/Re1250/fc0.3/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.3)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 2.5*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST12")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    
    file_path = "/data/Re1500/fc0.1/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 3*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST13")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    file_path = "/data/Re1500/fc0.2/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.2)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 3*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST14")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    file_path = "/data/Re1500/fc0.3/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.3)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 3*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST15")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    
    
    file_path = "/data/Re1750/fc0.1/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 3.5*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST16")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    file_path = "/data/Re1750/fc0.2/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.2)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 3.5*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST17")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    file_path = "/data/Re1750/fc0.3/comsol_data_transformed.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.3)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 3.5*0.3775)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "TEST18")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    file_path = "/data/Specific/SingleCaseComsol_trans.csv"
    if os.path.exists(to_absolute_path(file_path)):
        
        mapping = {"x": "x", "y": "y", "u ": "u", "v ": "v", "p ": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)    
        openfoam_var.update({"fc": np.full_like(openfoam_var["x"], 0.2)})
        openfoam_var.update({"inlet_u": np.full_like(openfoam_var["x"], 1)})   
            
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y","fc","inlet_u"]
        }    
            
        print("File found and processed.")   
            
            
        openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u", "v", "p",'warpx','warpy',"cline","radius"]
    )   
        domain.add_inferencer(openfoam_inferencer, "multicase")   
    else:
        print(f"File not found: {to_absolute_path(file_path)}")
    
    
    
    
    
    
    # make solver
    slv = Solver(cfg, domain)
    
    slv.solve()

if __name__ == "__main__":
    run()  
