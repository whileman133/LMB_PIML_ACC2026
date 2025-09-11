"""
util.py

Utilities for working with LMB cell models.

2024.10.26 | Created | Wesley Hileman <whileman@uccs.edu>
"""

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pybamm
import numpy as np
import scipy.io as sio
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator

from lumped import lmbcell


#
# LMB cell data model and utilities.
#

@dataclass
class Constants:
    QAh: float
    psi: float
    W: float
    vmin: float
    vmax: float
    Rc: float


@dataclass
class PorousElectrolyte:
    kappa: float
    tau: float


@dataclass
class OCP:
    theta: np.ndarray
    U: np.ndarray
    dU: np.ndarray
    # MSMR parameters
    U0: np.ndarray
    X: np.ndarray
    omega: np.ndarray


@dataclass
class Electrode3D(PorousElectrolyte):
    sigma: float
    Ds_ref: float
    theta0: float
    theta100: float
    ocp: OCP
    k0: np.ndarray
    beta: np.ndarray
    Rf: float
    Rdl: float
    Cdl: float


@dataclass
class Electrode2D:
    k0: float
    beta: float
    Rf: float
    Rdl: float
    Cdl: float


@dataclass
class LMBCell:
    cst: Constants
    neg: Electrode2D
    eff: PorousElectrolyte
    pos: Electrode3D


def get_ocp(cell: LMBCell, soc: float, TdegC: float = 25):
    """
    Solve MSMR model for Uocp in positive electrode given SOC and temperature.
    """

    U0 = cell.pos.ocp.U0
    X = cell.pos.ocp.X
    omega = cell.pos.ocp.omega
    f = pybamm.constants.F.value / pybamm.constants.R.value / (TdegC+273.15)
    pos_theta = cell.pos.theta0 + (soc / 100) * (cell.pos.theta100 - cell.pos.theta0)
    pos_Uocp = fsolve(lambda U: pos_theta - np.sum(X / (1 + np.exp(f * (U - U0) / omega))), np.mean(U0))[0]
    return pos_Uocp


@dataclass
class MSMR_OCP_Data:
    U0: np.ndarray
    X: np.ndarray
    omega: np.ndarray
    f: np.ndarray
    gj: np.ndarray
    xj: np.ndarray

    theta: np.ndarray
    Uocp: np.ndarray
    dUocp: np.ndarray


def get_ocp_curve(cell: LMBCell, TdegC: float = 25, npoints: int = 1000):
    """
    Calculate entire stoichiometry-vs-OCP curve for LMB cell.

    :param cell: LMB cell model.
    :param TdegC: Temperature at which to perform the calculation.
    :param npoints: Number of points for which to evalulate the curve.
    :return: MSMR_OCP_Data object
    """

    # Define OCP vector.
    vmin = cell.cst.vmin
    vmax = cell.cst.vmax
    Uocp = np.linspace(vmin, vmax, npoints)

    # Collect MSMR parameters (as column vectors).
    U0 = cell.pos.ocp.U0[:, np.newaxis]
    X = cell.pos.ocp.X[:, np.newaxis]
    omega = cell.pos.ocp.omega[:, np.newaxis]
    f = pybamm.constants.F.value / pybamm.constants.R.value / (TdegC + 273.15)

    # Compute stoichiometry vector.
    gj = np.exp(f * (Uocp - U0) / omega)
    xj = X / (1 + gj)
    theta = np.sum(xj, axis=0)

    # Compute differential OCP vector.
    dUocp = -1 / (f * np.sum(X / omega * gj / (1 + gj)**2, axis=0))

    return MSMR_OCP_Data(U0, X, omega, f, gj, xj, theta, Uocp, dUocp)


def get_ds_curve(cell: LMBCell, TdegC: float = 25):
    """
    Calculate entire stoichiometry-vs-solid-diffusion-coefficient curve for LMB cell.

    :param cell: LMB cell model.
    :param TdegC: Temperature at which to perform the calculation.
    :return: theta, Ds
    """

    ocp = get_ocp_curve(cell, TdegC)
    Dsref = cell.pos.Ds_ref
    Ds = -ocp.f * Dsref * ocp.theta * (1 - ocp.theta) * ocp.dUocp

    return ocp.theta, Ds


def get_exchange_current(cell: LMBCell, thetass: np.ndarray, thetae: np.ndarray, TdegC: float = 25):
    """
    Calculate exchange current at the given solid-surface stoichiometry and salt composition values.
    Also returns the solid-surface potential, Uss.

    :param cell: LMB cell model.
    :param thetass: np.ndarray of solid-surface stoichiometry values.
    :param TdegC: Temperature at which to perform the calculation.
    :return: i0 (vector), Uss (vector)
    """

    ocp = get_ocp_curve(cell, TdegC)
    J = len(cell.pos.ocp.U0)
    X = ocp.X.reshape(J, -1)  # ensure column vectors
    omega = ocp.omega.reshape(J, -1)
    k0 = cell.pos.k0.reshape(J, -1)
    beta = cell.pos.beta.reshape(J, -1)

    Uss = PchipInterpolator(
        np.flip(ocp.theta),   # flip for ascending order
        np.flip(ocp.Uocp),
        extrapolate=True
    )(thetass)
    xjss = PchipInterpolator(
        ocp.Uocp,
        ocp.xj,
        axis=1,
        extrapolate=True
    )(Uss)

    i0j = k0*xjss**(omega*beta)*(X - xjss)**(omega*(1 - beta))*thetae**(1 - beta)/(X/2)**omega
    i0 = np.sum(i0j, axis=0)

    return i0, Uss


#
# PyBaMM simulation utilities
#

def sim_profile(
    cell: LMBCell,
    time: np.ndarray,
    iapp: np.ndarray,
    soc0: float,
    TdegC: float = 25.0,
):
    """
    Solve the PDE model for the specified parameter values and current profile.

    :param cell: LMB cell model object containing parameter values.
    :param time: Vector of time values [s].
    :param iapp: Vector of applied current values [A].
    :param soc0: Initial cell SOC [%].
    :param TdegC: Cell temperature [degC]. DEFAULT 25.
    :return: PyBaMM Simulation object
    """

    J = len(cell.pos.ocp.U0)
    T = TdegC + 273.15
    pos_Uocp0 = get_ocp(cell, soc0, TdegC)

    # Construct PyBaMM model.
    pde_model = lmbcell.LumpedLMBModel(J=J)

    # Collect parameter values.
    param = get_param_values(cell)
    param.update({
        "T [K]": T,
        "iapp [A]": pybamm.Interpolant(
            time, iapp, pybamm.t, interpolator="linear"
        ),
        "pos_Uocp0 [V]": pos_Uocp0,
    }, check_already_exists=False)

    # Run simulation.
    sim = pybamm.Simulation(pde_model, parameter_values=param)
    sim.solve(t_eval=time)

    return sim


def sim_cc(
    cell: LMBCell,
    i_galv: float,
    soc0: float,
    socf: float,
    TdegC: float = 25.0,
    ts: float = 1.0,
    t_rest: float = 0,
):
    """
    Solve the PDE model for a constant-current dis/charge profile.

    :param cell:
    :param i_galv: Constant dis/charge current [A].
    :param soc0: Initial SOC [%].
    :param socf: Final SOC [%].
    :param TdegC: Cell temperature [degC]. DEFAULT 25.
    :param ts: Sampling interval [s]. DEFAULT 1.0.
    :return: PyBaMM Simulation object
    """

    i_galv = abs(i_galv)  # ignore sign, can infer from soc0 and socf

    # Total amount of charge to remove from (+) / add to (-) the cell [Ah].
    QdisAh = (soc0 - socf) / 100 * cell.cst.QAh

    # Total amount of time to spend dis/charging at i_galv [s].
    t_cc = 3600 * abs(QdisAh) / i_galv

    # Build time and applied current vectors.
    time = np.arange(0, t_cc + t_rest + ts, ts)
    iapp = np.zeros_like(time)
    iapp[time <= t_cc] = np.sign(soc0 - socf)*i_galv

    # Run the simulation.
    return sim_profile(cell, time, iapp, soc0, TdegC)


def sim_gitt(
    cell: LMBCell,
    t_galv: float,
    t_rest: float,
    i_galv: float,
    soc0: float,
    socf: float,
    TdegC: float = 25.0,
    ts: float = 1.0,
):
    """
    Solve the PDE model for a galvanostatic intermittent titration
    technique (GITT) current profile.

    :param cell: LMB cell model object containing parameter values.
    :param t_galv: Dis/charge interval duration [s].
    :param t_rest: Rest interval duration [s].
    :param i_galv: Current magnitude during dis/charge intervals [A].
    :param soc0: Initial SOC of the cell [%].
    :param socf: Final SOC of the cell [%].
    :param TdegC: Temperature of the cell [degC]. DEFAULT 25.
    :param ts: Sampling interval [s]. DEFAULT 1.0.
    :return: PyBaMM Simulation object
    """

    i_galv = abs(i_galv)  # ignore sign, can infer from soc0 and socf

    # Total amount of charge to remove from (+) / add to (-) the cell [Ah].
    QdisAh = (soc0 - socf)/100 * cell.cst.QAh

    # Total amount of time to spend dis/charging at i_galv [s].
    t_dis = 3600 * abs(QdisAh) / i_galv

    # Number of required dis/charge intervals.
    n_dis = math.ceil(t_dis/t_galv)

    # Total duration of the GITT profile.
    t_gitt = n_dis * (t_galv + t_rest)

    # Build time and applied current vectors.
    time = np.arange(0, t_gitt + ts, ts)
    iapp = np.zeros_like(time)
    for k in range(n_dis):
        t_start = k * (t_galv + t_rest)
        ind = np.logical_and(t_start <= time, time <= t_start + t_galv)
        iapp[ind] = np.sign(soc0 - socf) * i_galv

    # Run the simulation.
    return sim_profile(cell, time, iapp, soc0, TdegC)


def sim_drive(
    cell: LMBCell,
    cycle_name: str,
    soc0: float,
    socf: float,
    TdegC: float = 25.0,
    ts: float = 1.0,
):
    """
    Solve the PDE model for a dynamic drive current profile.

    :param cell: LMB cell model.
    :param cycle_name: Name of the drive cycle ('UDDS', 'LA92', or 'US06')
    :param soc0: Initial cell soc [%].
    :param socf: Final cell soc [%].
    :param TdegC: Cell temperature. DEFAULT 25.
    :param ts: Sampling interval [s]. DEFAULT 1.0.
    :return:
    """

    # Load drive cycle from file.
    data = np.genfromtxt(os.path.join('drive_cycles', f"{cycle_name}.csv"), delimiter=',')
    time0 = data[:, 0]
    iapp0 = data[:, 1]

    # Interpolate over unform time variable.
    time = np.arange(np.min(time0), np.max(time0) + ts, ts)
    iapp = PchipInterpolator(time0, iapp0)(time)

    # Rescale applied current.
    QdisAh0 = np.trapz(iapp, time)/3600
    QdisAh = (soc0 - socf)/100 * cell.cst.QAh
    iapp = iapp*QdisAh/QdisAh0

    # Run the simulation.
    return sim_profile(cell, time, iapp, soc0, TdegC)


def get_param_values(cell: LMBCell):
    """
    Construct PyBaMM parameter values for an LMB cell object.
    """

    J = len(cell.pos.ocp.U0)

    # Collect parameter values.
    msmr_param = {}
    for j in range(J):
        msmr_param[f"pos_U0_{j} [V]"] = cell.pos.ocp.U0[j]
        msmr_param[f"pos_X_{j}"] = cell.pos.ocp.X[j]
        msmr_param[f"pos_omega_{j}"] = cell.pos.ocp.omega[j]
        msmr_param[f"pos_k0_{j} [A]"] = cell.pos.k0[j]
        msmr_param[f"pos_beta_{j}"] = cell.pos.beta[j]
    return pybamm.ParameterValues({
        "vmin [V]": cell.cst.vmin,
        "vmax [V]": cell.cst.vmax,
        "W": cell.cst.W,
        "psi [V]": cell.cst.psi,
        "Q [Ah]": cell.cst.QAh,
        "pos_sigma [Ohm-1]": cell.pos.sigma,
        "pos_Dsref [s-1]": cell.pos.Ds_ref,
        "pos_kappa [Ohm-1]": cell.pos.kappa,
        "pos_taue [s]": cell.pos.tau,
        "pos_theta0": cell.pos.theta0,
        "pos_theta100": cell.pos.theta100,
        "pos_Rf [Ohm]": cell.pos.Rf,
        "pos_Rdl [Ohm]": cell.pos.Rdl,
        "pos_Cdl [F]": cell.pos.Cdl,
        **msmr_param,
        "eff_kappa [Ohm-1]": cell.eff.kappa,
        "eff_taue [s]": cell.eff.tau,
        "neg_k0 [A]": cell.neg.k0,
        "neg_beta": cell.neg.beta,
        "neg_Rdl [Ohm]": cell.neg.Rdl,
        "neg_Cdl [F]": cell.neg.Cdl,
    })


#
# MATLAB io utilities.
#

def load_mat_cell_model(filepath):
    """
    Load cell model parameters from .mat file.
    """

    cell_data = sio.loadmat(filepath)

    # Fetch Ds-versus-theta lookup table.
    theta = cell_data['dataDs']['theta'][0, 0].T[0]
    Uocp = cell_data['dataDs']['Uocp'][0, 0].T[0]
    dUocp = cell_data['dataDs']['dUocp'][0, 0].T[0]
    # Ds = cell_data['dataDs']['Ds'][0, 0].T[0]

    # Fetch cell parameters and construct cell model.
    cellParams = cell_data['cellParams']
    cst = Constants(
        QAh=cellParams['const'][0, 0]['Q'][0, 0][0, 0],
        psi=cellParams['const'][0, 0]['psi'][0, 0][0, 0],
        W=cellParams['const'][0, 0]['W'][0, 0][0, 0],
        vmin=cellParams['const'][0, 0]['Vmin'][0, 0][0, 0],
        vmax=cellParams['const'][0, 0]['Vmax'][0, 0][0, 0],
        Rc=cellParams['pkg'][0, 0]['R0'][0, 0][0, 0])
    neg = Electrode2D(
        k0=cellParams['neg'][0, 0]['k0'][0, 0][0, 0],
        beta=cellParams['neg'][0, 0]['alpha'][0, 0][0, 0],
        Rf=cellParams['neg'][0, 0]['Rf'][0, 0][0, 0],
        Rdl=cellParams['neg'][0, 0]['Rdl'][0, 0][0, 0],
        Cdl=cellParams['neg'][0, 0]['Cdl'][0, 0][0, 0])
    eff = PorousElectrolyte(
        kappa=cellParams['eff'][0, 0]['kappa'][0, 0][0, 0],
        tau=cellParams['eff'][0, 0]['tauW'][0, 0][0, 0])
    pos = Electrode3D(
        sigma=cellParams['pos'][0, 0]['sigma'][0, 0][0, 0],
        kappa=cellParams['pos'][0, 0]['kappa'][0, 0][0, 0],
        tau=cellParams['pos'][0, 0]['tauW'][0, 0][0, 0],
        Ds_ref=cellParams['pos'][0, 0]['Dsref'][0, 0][0, 0],
        theta0=cellParams['pos'][0, 0]['theta0'][0, 0][0, 0],
        theta100=cellParams['pos'][0, 0]['theta100'][0, 0][0, 0],
        ocp=OCP(
            theta, Uocp, dUocp,
            U0=cellParams['pos'][0, 0]['U0'][0, 0].T[0],
            X=cellParams['pos'][0, 0]['X'][0, 0].T[0],
            omega=cellParams['pos'][0, 0]['omega'][0, 0].T[0],
        ),
        k0=cellParams['pos'][0, 0]['k0'][0, 0].T[0],
        beta=cellParams['pos'][0, 0]['alpha'][0, 0].T[0],
        Rf=cellParams['pos'][0, 0]['Rf'][0, 0][0, 0],
        Rdl=cellParams['pos'][0, 0]['Rdl'][0, 0][0, 0],
        Cdl=cellParams['pos'][0, 0]['Rdl'][0, 0][0, 0])
    cell = LMBCell(cst, neg, eff, pos)

    return cell


@dataclass
class FOMOutput:
    soc0Pct: float
    ts: float
    time: np.ndarray
    iapp: np.ndarray
    vcell: np.ndarray
    thetass: np.ndarray       # over pos electrode
    thetass_cc: np.ndarray    # at pos current-collector
    thetae: np.ndarray
    xthetass: np.ndarray
    xthetae: np.ndarray


def load_mat_comsol_sim(filepath):
    """
    Load COMSOL simulation output from .mat file.
    """

    simData = sio.loadmat(filepath)
    soc0Pct = simData['simData']['arg'][0, 0]['soc0Pct'][0, 0][0, 0]
    time = simData['simData']['time'][0, 0].T[0]
    ts = float(np.mean(np.diff(time)))
    iapp = simData['simData']['iapp'][0, 0].T[0]
    vcell = simData['simData']['vcell'][0, 0].T[0]
    thetass = simData['simData']['output'][0, 0]['Thetass'][0, 0]
    thetass_cc = thetass[:, -1]
    thetae = simData['simData']['output'][0, 0]['Thetae'][0, 0]
    xthetass = simData['simData']['output'][0, 0]['xLocs'][0, 0]['Thetass'][0, 0].T[0]
    xthetae = simData['simData']['output'][0, 0]['xLocs'][0, 0]['Thetae'][0, 0].T[0]

    # COMSOL model uses dll, sep, and pos layers, but dll and sep have identical properties
    # we use the eff layer approximation.
    # Shift indices for a cell with eff and pos layers only.
    xthetass -= 1
    ind02 = np.logical_and(0 <= xthetae, xthetae <= 2)
    ind23 = np.logical_and(2 < xthetae, xthetae <= 3)
    xthetae[ind02] /= 2
    xthetae[ind23] -= 1

    return FOMOutput(soc0Pct, ts, time, iapp, vcell, thetass, thetass_cc, thetae, xthetass, xthetae)


@dataclass
class ROMOutput:
    time: np.ndarray
    iapp: np.ndarray
    vcell: np.ndarray


def load_mat_rom_sim(filepath):
    """
    Load ROM simulation output from .mat file.
    """

    simData = sio.loadmat(filepath)
    time = simData['ROMout']['time'][0, 0].T[0]
    iapp = simData['simData']['Iapp'][0, 0].T[0]
    vcell = simData['ROMout']['Vcell'][0, 0].T[0]

    return ROMOutput(time, iapp, vcell)


#
# ML utilities.
#

class MuSigmaNormalizer:
    """
    Scale a time-indexed dataset by the mean and standard deviation
    of each input/output feature.

    Assumes dim0=sample, dim1=time, dim2=feature
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # Note: use nanmean/nanstd to ignore nans, which may be present due to padding.
        self.muX = np.nanmean(X, axis=(0, 1))
        self.sigX = np.nanstd(X, axis=(0, 1))
        self.muY = np.nanmean(Y, axis=(0, 1))
        self.sigY = np.nanstd(Y, axis=(0, 1))
        self.Xs = (X - self.muX) / self.sigX
        self.Ys = (Y - self.muY) / self.sigY

    def get_values(self):
        return self.Xs, self.Ys

    def pack_x(self, X: np.ndarray):
        Xs = (X - self.muX) / self.sigX
        return Xs

    def pack_y(self, Y: np.ndarray):
        Ys = (Y - self.muY) / self.sigY
        return Ys

    def unpack_y(self, Ys: np.ndarray):
        Y = Ys*self.sigY + self.muY
        return Y

    def wrap(self, fn, *args, **kwargs):
        def wrapper(X):
            Xs = self.pack_x(X)
            Ys = fn(Xs, *args, **kwargs)
            Y = self.unpack_y(Ys)
            return Y
        return wrapper


#
# Filter utilities.
#


def iir_1p(X: np.ndarray, tau: float, ts: float):
    """
    Compute response of a single-parameter discrete-time IIR filter.
    :param X: Input vector.
    :param tau: Time constant [s].
    :param ts: Sampling interval [s].
    :return: Output vector Y.
    """
    a = math.exp(-ts/tau)
    Y = np.zeros_like(X)
    y = 0
    for k in range(1, len(X)):
        y = a*y + (1 - a)*X[k - 1]
        Y[k] = y
    return Y


def iir_2p(X: np.ndarray, a: float, b: float):
    """
    Compute response of a two-parameter discrete-time IIR filter.
    :param X: Input vector.
    :param tau: Time constant [s].
    :param ts: Sampling interval [s].
    :return: Output vector Y.
    """
    Y = np.zeros_like(X)
    y = 0
    for k in range(1, len(X)):
        y = a*y + b*X[k - 1]
        Y[k] = y
    return Y


#
# Data preparation utilities.
#


def sim_specs(series_key, sim_data):
    if series_key == 'cc':
        return f"{sim_data['C_rate']}C"
    elif series_key == 'gitt':
        return f"{sim_data['t_galv']/60:.0f}m-{sim_data['t_rest']/60:.0f}m-{sim_data['C_rate']}C"
    elif series_key == 'drive':
        return f"{sim_data['cycle_name']}"


class BaseFNNAdapter(ABC):
    def __init__(self, tau, ts, include_delta=True):
        if not isinstance(tau, (list, tuple, np.ndarray)):
            tau = [tau]
        self.tau = tau
        self.ts = ts
        self.include_delta = include_delta

    def get_delta(self, iapp):
        delta_shape = len(self.tau), len(iapp)
        delta = np.zeros(delta_shape)
        for idx, tau in enumerate(self.tau):
            delta[idx,:] = iir_1p(iapp, tau, self.ts)
        return delta

    @abstractmethod
    def pack_x(self, *args, **kwargs):
        pass

    @abstractmethod
    def pack_y(self, *args, **kwargs):
        pass

    @abstractmethod
    def unpack_y(self, *args, **kwargs):
        pass

    def wrap(self, fn, *args, **kwargs):
        """
        Wraps a given function with additional functionality to process input and output
        data. This method takes a function and returns a new function that preprocesses
        the input using the `pack_x` method, invokes the original function, and then
        postprocesses the output using the `unpack_y` method.

        :param fn: The function to be wrapped.
        :param args: Positional arguments to be passed to the wrapped function.
        :param kwargs: Keyword arguments to be passed to the wrapped function.
        :return: A new function that preprocesses input, processes it via the wrapped
                 function, and postprocesses the output.
        """
        def wrapper(input):
            X = self.pack_x(input)
            Y = fn(X, *args, **kwargs)
            output = self.unpack_y(Y)
            return output
        return wrapper


class ThetassFNNAdapter(BaseFNNAdapter):
    """
    Utility class for encoding/decoding data for FNN that predicts thetass2.
    """

    def pack_x(self, spm_state):
        iapp = spm_state['iapp']
        delta = self.get_delta(iapp)
        thetass = spm_state['thetass']
        thetas_avg = spm_state['thetas_avg']

        if self.include_delta:
            x = np.vstack((thetass, thetas_avg, delta)).transpose()
        else:
            x = np.vstack((thetass, thetas_avg)).transpose()

        return x

    def pack_y(self, ground_truth):
        thetass2_true = ground_truth['thetass2']
        y = np.stack((thetass2_true,), axis=-1)  # make column vector
        return y

    def unpack_y(self, y: np.ndarray):
        if y.ndim == 1:
            thetass2 = y
        else:
            thetass2, = [np.squeeze(a) for a in np.split(y, y.shape[-1], axis=-1)]
        thetass2 = np.clip(thetass2, 0, 1)
        return {
            'thetass2': thetass2,
        }


class PhieFNNAdapter(BaseFNNAdapter):
    """
    Utility class for encoding/decoding data for FNN that predicts phie2.
    """

    def pack_x(self, spm_state):
        iapp = spm_state['iapp']
        delta = self.get_delta(iapp)
        thetae0 = spm_state['thetae0']
        thetae2 = spm_state['thetae2']
        thetass = spm_state['thetass']
        thetas_avg = spm_state['thetas_avg']
        if self.include_delta:
            x = np.vstack((thetae0, thetae2, thetass, thetas_avg, iapp, delta)).transpose()
        else:
            x = np.vstack((thetae0, thetae2, thetass, thetas_avg, iapp)).transpose()
        return x

    def pack_y(self, ground_truth):
        phie2_true = ground_truth['phie2']
        y = np.stack((phie2_true,), axis=-1)
        return y

    def unpack_y(self, y: np.ndarray):
        if y.ndim == 1:
            phie2 = y
        else:
            phie2, = [np.squeeze(a) for a in np.split(y, y.shape[-1], axis=-1)]
        return {
            'phie2': phie2,
        }


class IfFNNAdapter(BaseFNNAdapter):
    """
    Utility class for encoding/decoding data for FNN that predicts phie2.
    """

    def pack_x(self, spm_state):
        iapp = spm_state['iapp']
        delta = self.get_delta(iapp)
        thetae0 = spm_state['thetae0']
        thetae2 = spm_state['thetae2']
        thetass = spm_state['thetass']
        thetas_avg = spm_state['thetas_avg']
        if self.include_delta:
            x = np.vstack((thetae0, thetae2, thetass, thetas_avg, iapp, delta)).transpose()
        else:
            x = np.vstack((thetae0, thetae2, thetass, thetas_avg, iapp)).transpose()
        return x

    def pack_y(self, ground_truth):
        if2_true = ground_truth['if2']
        y = np.stack((if2_true,), axis=-1)
        return y

    def unpack_y(self, y: np.ndarray):
        if y.ndim == 1:
            if2 = y
        else:
            if2, = [np.squeeze(a) for a in np.split(y, y.shape[-1], axis=-1)]
        return {
            'if2': if2,
        }


class IfPhieFNNAdapter(BaseFNNAdapter):
    """
    Utility class for encoding/decoding data for FNN that predicts if2 and phie2.
    """

    def pack_x(self, spm_state):
        iapp = spm_state['iapp']
        delta = iir_1p(iapp, self.tau, self.ts)
        thetae0 = spm_state['thetae0']
        thetae2 = spm_state['thetae2']
        thetass = spm_state['thetass']
        thetas_avg = spm_state['thetas_avg']
        if self.include_delta:
            x = np.stack((thetae0, thetae2, thetass, thetas_avg, iapp, delta), axis=-1)
        else:
            x = np.stack((thetae0, thetae2, thetass, thetas_avg, iapp), axis=-1)
        return x

    def pack_y(self, ground_truth):
        if2_true = ground_truth['if2']
        phie2_true = ground_truth['phie2']
        y = np.stack((if2_true, phie2_true), axis=-1)
        return y

    def unpack_y(self, y: np.ndarray):
        if2, phie2 = [np.squeeze(a) for a in np.split(y, y.shape[-1], axis=-1)]
        return {
            'if2': if2,
            'phie2': phie2,
        }