"""
spm.py

Single-particle model for LMB cell.

2024.09.15 | Created | Wesley Hileman <whileman@uccs.edu>
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pybamm
import scipy.constants
from scipy.interpolate import CubicSpline, PchipInterpolator

import util
from util import LMBCell


@dataclass
class FVSData:
    """
    Data output from finite-volume simulation.
    """

    edges: np.ndarray   # vector of volume edge locations
    X: np.ndarray       # Nsim x Nstates matrix of states
    ts: float           # Sampling interval [s]
    iapp: np.ndarray    # applied current vector [A]


class BaseFVS(ABC):
    """
    Abstract base class for finite-volume simulations (FVS).
    """

    N: int      # total number of states (volumes)
    ts: float   # sampling interval [s]
    edges: np.ndarray    # vector of volume edge locations (x or r)
    centers: np.ndarray  # vector of volume center points (x or r)

    # Continuous-time state-space matrices.
    A: np.ndarray
    B: np.ndarray

    # Discrete-time state-space matrices.
    Ad: np.ndarray
    Bd: np.ndarray

    # Stateful simulation properties.
    x: np.ndarray   # current state vector

    def __init__(self):
        self.x = np.array([])

    @abstractmethod
    def get_initial(self, **kwargs):
        pass

    def get_ss_matrices(self, x: np.ndarray):
        """
        Fetch state-space matrices given the present state.
        """

        # Default: time invariant.
        return self.Ad, self.Bd

    def run(self, iapp: np.ndarray, **kwargs) -> FVSData:
        """
        Run the finite-volume simulation given the applied current vector.
        Any additional keyword arguments will be passed to get_initial() to determine the
        initial state.
        """

        N = self.N        # number of states
        kmax = len(iapp)  # number of simulation steps

        # Initialize storage.
        X = np.zeros(shape=(kmax, N))

        # Initialize state vector.
        x = self.get_initial(**kwargs)

        X[0, :] = x.T
        for k in range(1, kmax):
            Ad, Bd = self.get_ss_matrices(x)   # ss matrices for this iteration
            x = Ad @ x + Bd * iapp[k - 1]
            X[k, :] = x.T

        return FVSData(self.edges, X, self.ts, iapp)

    def initialize(self, **kwargs):
        """
        Initialize the finite volume simulation.
        Any additional keyword arguments will be passed to get_initial() to determine the
        initial state.
        Returns the initial state vector.
        """

        # Initialize state vector.
        self.x = self.get_initial(**kwargs)
        self.Ad, self.Bd = self.get_ss_matrices(self.x)

        return self.x

    def step(self, iappk: float):
        """
        Run a single time-step of the FVS.
        Returns the new state vector.
        """
        self.x = self.Ad @ self.x + self.Bd * iappk
        self.Ad, self.Bd = self.get_ss_matrices(self.x)  # ss matrices for next iteration
        return self.x


class ElectrolyteFVS(BaseFVS):
    """
    Finite-volume simulation for electrolyte in an LMB cell.
    """

    def __init__(self, cell: LMBCell, Ne: int, Np: int, ts: float = 1.0, TdegC: float = 25):
        super().__init__()
        self.cell = cell
        self.Ne = Ne
        self.Np = Np
        self.N = Ne + Np
        self.ts = ts
        self.TdegC = TdegC

        # Fetch relevant parameters.
        psi = cell.cst.psi
        ke = cell.eff.kappa
        te = cell.eff.tau
        kp = cell.pos.kappa
        tp = cell.pos.tau
        T = TdegC+273.15  # to Kelvin
        dxe = 1/Ne
        dxp = 1/Np
        dxie = (kp*dxe + ke*dxp)/2/kp
        dxip = (kp*dxe + ke*dxp)/2/ke

        # Build state-space matrices (time invariant) ----------------------------------------------
        A = np.zeros(shape=(Ne+Np, Ne+Np), dtype=np.float64)
        B = np.zeros(shape=(Ne+Np, 1), dtype=np.float64)

        # Eff layer contribution.
        for ssi, i in enumerate(range(1, Ne+1)):
            if i == 1:
                A[ssi, [ssi+1, ssi]] = np.array(
                    [1, -1]
                )/te/dxe/dxe
                B[ssi] = 1/te/ke/psi/T/dxe
            elif i == Ne:
                A[ssi, [ssi+1, ssi, ssi-1]] = np.array(
                    [1/dxie/dxe, -(1/dxie/dxe + 1/dxe/dxe), 1/dxe/dxe]
                )/te
            else:
                A[ssi, [ssi+1, ssi, ssi-1]] = np.array(
                    [1, -2, 1]
                )/te/dxe/dxe

        # Pos layer contribution.
        for ssi, i in enumerate(range(1, Np+1), start=Ne):
            B[ssi] = -1/tp/kp/psi/T
            if i == 1:
                A[ssi, [ssi+1, ssi, ssi-1]] = np.array(
                    [1/dxp/dxp, -(1/dxp/dxp + 1/dxip/dxp), 1/dxip/dxp]
                )/tp
            elif i == Np:
                A[ssi, [ssi, ssi-1]] = np.array(
                    [-1, 1]
                )/tp/dxp/dxp
            else:
                A[ssi, [ssi+1, ssi, ssi-1]] = np.array(
                    [1, -2, 1]
                )/tp/dxp/dxp

        # Convert to discrete time using backward Euler method.
        Ad = np.linalg.inv(np.identity(Ne+Np) - ts*A)
        Bd = Ad@(ts*B)

        # Determine edges of the volumes.
        xeff = np.linspace(0, 1, Ne+1)
        xpos = np.linspace(1, 2, Np+1)
        edges = np.concatenate((xeff[:-1], xpos))

        # Determine center points of the volumes.
        centers = edges[:-1] + np.diff(edges) / 2

        self.A = A
        self.B = B
        self.Ad = Ad
        self.Bd = Bd
        self.edges = edges
        self.centers = centers

    def get_initial(self, **kwargs):
        # Begin the simulation fully equilibrated.
        return np.ones(shape=(self.N, 1))

    def thetae(self, x, Xe=None):
        """
        Evaluate the electrolyte stoichiometry. Cubic spline interpolation over radial position.

        :param x: Linear position(s) at which to evaluate thetae. Float or numpy array.
        :param Xe: State matrix for the FVS. If None, uses the current state. Default None.
        :return: thetae
        """
        if Xe is None:
            Xe = self.x.T

        if isinstance(x, np.ndarray):
            # Force row vector.
            x = x.reshape(-1)

        return PchipInterpolator(self.centers, Xe, extrapolate=True, axis=1)(x)


class SolidFVS(BaseFVS):
    """
    Finite-volume simulation in spherical solid particle for LMB cell.
    """

    def __init__(self, cell: LMBCell, N: int, ts: float = 1.0, TdegC: float = 25, use_constant_Ds=True):
        """

        :param cell: Cell model containing parameter values.
        :param N: Number of shells to employ in FVS.
        :param ts: Sampling interval [s].
        :param TdegC: Temperature for computing log-average diffusivity.
        """

        super().__init__()
        self.cell = cell
        self.N = N
        self.ts = ts
        self.use_constant_Ds = use_constant_Ds

        theta_vect, Ds_vect = util.get_ds_curve(cell, TdegC)
        ind = np.argsort(theta_vect)
        theta_vect = theta_vect[ind]
        Ds_vect = Ds_vect[ind]
        self.theta = theta_vect
        self.Ds = Ds_vect

        # Determine edges of the volumes.
        self.edges = np.linspace(0, 1, N + 1)

        # Determine center points of the volumes.
        self.centers = self.edges[:-1] + np.diff(self.edges) / 2

        # Precompute SS matrices if we're using constant diffusivity.
        if use_constant_Ds:
            # Use log-average diffusivity.
            Ds = 10 ** np.mean(np.log10(Ds_vect))
            QAh = cell.cst.QAh
            theta0 = cell.pos.theta0
            theta100 = cell.pos.theta100

            # Compute constant state-space matricies.
            A = np.zeros(shape=(N, N), dtype=np.float64)
            B = np.zeros(shape=(N, 1), dtype=np.float64)
            for ssi, i in enumerate(range(1, N + 1)):
                if i == 1:
                    A[ssi, [ssi + 1, ssi]] = np.array(
                        [1, -1]
                    ) * 3 * Ds * N ** 2
                elif i == N:
                    A[ssi, [ssi, ssi - 1]] = np.array(
                        [-1, +1]
                    ) * 3 * Ds * N ** 2 * (N - 1) ** 2 / (N ** 3 - (N - 1) ** 3)
                    B[ssi] = np.abs(theta100 - theta0) * N ** 3 / (N ** 3 - (N - 1) ** 3) / 3600 / QAh
                else:
                    A[ssi, [ssi + 1, ssi, ssi - 1]] = np.array(
                        [Ds * i ** 2, -(Ds * i ** 2 + Ds * (i - 1) ** 2), Ds * (i - 1) ** 2]
                    ) * 3 * N ** 2 / (i ** 3 - (i - 1) ** 3)

            # Convert to discrete time using backward Euler method.
            Ad = np.linalg.inv(np.identity(N) - ts * A)
            Bd = Ad @ (ts * B)

            self.Ad = Ad
            self.Bd = Bd

    def get_initial(self, **kwargs):
        # Begin the simulation fully equilibrated at the given SOC.
        socPct0 = kwargs['soc0_pct']
        theta0 = self.cell.pos.theta0
        theta100 = self.cell.pos.theta100
        thetas0 = theta0 + (socPct0 / 100) * (theta100 - theta0)
        return np.ones(shape=(self.N, 1))*thetas0

    def get_ss_matrices(self, x: np.ndarray):
        if self.use_constant_Ds:
            return self.Ad, self.Bd

        # Otherwise, build time-variant matrices:

        cell = self.cell
        N = self.N
        ts = self.ts
        QAh = cell.cst.QAh
        theta0 = cell.pos.theta0
        theta100 = cell.pos.theta100

        # Fetch solid diffusivity at the edges of the shells.
        thetas_edges = self.thetas(self.edges, x.reshape(-1))  # force x row vector
        Ds_edges = 10**PchipInterpolator(self.theta, np.log10(self.Ds), extrapolate=True)(thetas_edges)

        # Built continuous time matrices.
        A = np.zeros(shape=(N, N), dtype=np.float64)
        B = np.zeros(shape=(N, 1), dtype=np.float64)
        for ssi, i in enumerate(range(1, N + 1)):
            if i == 1:
                Ds = Ds_edges[1]
                A[ssi, [ssi + 1, ssi]] = np.array(
                    [1, -1]
                ) * 3 * Ds * N ** 2
            elif i == N:
                Ds = Ds_edges[N - 1]
                A[ssi, [ssi, ssi - 1]] = np.array(
                    [-1, +1]
                ) * 3 * Ds * N ** 2 * (N - 1) ** 2 / (N ** 3 - (N - 1) ** 3)
                B[ssi] = np.abs(theta100 - theta0) * N ** 3 / (N ** 3 - (N - 1) ** 3) / 3600 / QAh
            else:
                Ds_i = Ds_edges[i]
                Ds_i_1 = Ds_edges[i - 1]
                A[ssi, [ssi + 1, ssi, ssi - 1]] = np.array(
                    [Ds_i * i ** 2, -(Ds_i * i ** 2 + Ds_i_1 * (i - 1) ** 2), Ds_i_1 * (i - 1) ** 2]
                ) * 3 * N ** 2 / (i ** 3 - (i - 1) ** 3)

        # Convert to discrete time using backward Euler method.
        Ad = np.linalg.inv(np.identity(N) - ts * A)
        Bd = Ad @ (ts * B)

        return Ad, Bd

    def thetas(self, r, Xs=None):
        """
        Evaluate the solid stoichiometry. Pchip interpolation over radial position.

        :param r: Radial position(s) at which to evaluate thetas. Float or numpy array.
        :param Xs: State matrix for the FVS. If None, uses the current state. Default None.
        :return: thetas
        """
        if Xs is None:
            Xs = self.x.T

        if isinstance(r, np.ndarray):
            # Force row vector.
            r = r.reshape(-1)

        return PchipInterpolator(self.centers, Xs, extrapolate=True, axis=1)(r)

    def thetas_avg(self, Xs=None):
        """
        Calculate the average solid stoichiometry in the spherical particle.

        :param Xs: State matrix for the FVS. If None, uses the current state. Default None.
        :return: thetas_avg
        """

        if Xs is None:
            Xs = self.x.T

        # Compute sum(theta_i * V_i) / V_total where:
        #   V_i = volume of ith shell = (4/3)*pi*(r[i]^3 - r[i-1]^3)
        #   V_total = volume of unit sphere = (4/3)*pi*(1^3)
        return np.sum(Xs * np.diff(self.edges**3), axis=1)


@dataclass
class SPMeOutput:
    Xe: np.ndarray
    Xs: np.ndarray
    iapp: np.ndarray
    vcell: np.ndarray
    thetae0: np.ndarray
    thetae2: np.ndarray
    thetass: np.ndarray
    thetas_avg: np.ndarray
    phie2: np.ndarray
    neg_phis: np.ndarray
    pos_phise: np.ndarray
    pos_etas: np.ndarray
    pos_Uss: np.ndarray
    pos_if: np.ndarray

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class SPMe:
    """
    Implements single-particle model with electrolyte for an LMB cell.
    """

    def __init__(self, cell: LMBCell, ns: int, ne_eff: int, ne_pos: int, ts: float = 1.0, TdegC: float = 25):
        self.cell = cell
        self.ts = ts
        self.TdegC = TdegC
        self.electrolyte_fvs = ElectrolyteFVS(cell, Ne=ne_eff, Np=ne_pos, ts=ts, TdegC=TdegC)
        self.solid_fvs = SolidFVS(cell, N=ns, ts=ts, TdegC=TdegC)

    def run(self, iapp: np.ndarray, soc0_pct: float, **kwargs) -> SPMeOutput:
        """
        Simulate the SPMe given the applied current vector.
        """

        Ne = self.electrolyte_fvs.N  # number of electrolyte states
        Ns = self.solid_fvs.N        # number of solid states
        kmax = len(iapp)             # number of simulation steps

        # Initialize storage.
        Xe = np.zeros(shape=(kmax, Ne))  # electrolyte state
        Xs = np.zeros(shape=(kmax, Ns))  # solid state

        # Initialize state vectors.
        xe = self.electrolyte_fvs.initialize()
        xs = self.solid_fvs.initialize(soc0_pct=soc0_pct)

        # Run simulation.
        Xe[0, :] = xe.T   # initial electrolyte state
        Xs[0, :] = xs.T   # initial solid state
        for k in range(1, kmax):
            xe = self.electrolyte_fvs.step(iapp[k - 1])
            xs = self.solid_fvs.step(iapp[k - 1])
            Xe[k, :] = xe.T
            Xs[k, :] = xs.T

        return self.get_output_variables(Xe, Xs, iapp, **kwargs)

    def get_output_variables(self, Xe, Xs, iapp, correction_fn=None) -> SPMeOutput:
        """
        Fetch output variables given state matrices.
        """

        # Collect cell parameter values.
        W = self.cell.cst.W
        psi = self.cell.cst.psi
        T = self.TdegC + 273.15
        ke = self.cell.eff.kappa
        kp = self.cell.pos.kappa
        k0n = self.cell.neg.k0
        betan = self.cell.neg.beta
        Rfp = self.cell.pos.Rf
        f = pybamm.constants.F.value / pybamm.constants.R.value / T

        # Fetch solid and electrolyte concentration variables
        # (matrices with dim0=time and dim1=position).
        thetae = self.electrolyte_fvs.thetae(x=np.array([0, 2]), Xe=Xe)
        thetae0 = thetae[:, 0]
        thetae2 = thetae[:, 1]
        thetass = self.solid_fvs.thetas(r=1, Xs=Xs)
        thetas_avg = self.solid_fvs.thetas_avg(Xs=Xs)

        # Calculate phie2.
        phie2 = -iapp*(1/ke + 1/kp/2) + W*psi*T*(thetae2 - thetae0)

        # Assign Faradaic currents.
        ifn = iapp
        ifp = -iapp

        # Correct predictions if a correction function is supplied.
        if correction_fn is not None:
            corrections = correction_fn({
                'iapp': iapp,
                'thetass': thetass,
                'thetas_avg': thetas_avg,
                'thetae0': thetae0,
                'thetae2': thetae2,
            })
            if 'thetass2' in corrections:
                thetass = corrections['thetass2']
            if 'if2' in corrections:
                ifp = corrections['if2']
            if 'phie2' in corrections:
                phie2 = corrections['phie2']

        # Calculate exchange currents and solid-surface potential at pos.
        i0n = k0n * thetae0 ** (1 - betan)
        i0p, Uss = util.get_exchange_current(self.cell, thetass, thetae2, self.TdegC)

        # Calculate eta_s(neg)=phis(neg)
        # (assumes beta_n=0.5).
        phis_n = 2/f*np.arcsinh(ifn/i0n/2)

        # Calculate eta_s(pos) and phise(pos)
        # (assumes all beta_p=0.5).
        eta_p = 2/f*np.arcsinh(ifp/i0p/2)
        phise_p = Uss + eta_p + Rfp*ifp

        # Calculate cell voltage.
        vcell = phise_p + phie2 - phis_n

        # Collect output.
        return SPMeOutput(
            Xe=Xe,
            Xs=Xs,
            iapp=iapp,
            vcell=vcell,
            thetae0=thetae0,
            thetae2=thetae2,
            thetass=thetass,
            thetas_avg=thetas_avg,
            phie2=phie2,
            neg_phis=phis_n,
            pos_phise=phise_p,
            pos_etas=eta_p,
            pos_Uss=Uss,
            pos_if=ifp,
        )
