"""
libcell.py

PyBaMM partial-differential-equation (PDE) model for LIB cell.

2025.01.27 | Created | Wesley Hileman <whileman@uccs.edu>
"""

import numpy as np
import pybamm
from .base import BaseLumpedModel


class LumpedLIBModel(BaseLumpedModel):
    """
    Model for a lithium-ion battery cell with neg, sep, and pos layers.
    Lumped parameters. Dimensionless geometry. Butler-Volmer kinetics.
    Constant Ds. Standard OCP.

    NOTE 1: For simplicity, we set the potential reference point to the electrolyte at the surface
            of the Li-metal electrode, i.e., we define phi_e(0) = 0V. This makes phi_s_neg(0) nonzero,
            and it must be solved for to compute the cell voltage, thus phi_s_neg(0) appears as
            a variable of the model.

    NOTE 2: The double-layer models break when Rdl(pos)=0 or Rdl(neg)=0. Use small values instead.

    NOTE 3: You must specify the number of MSMR galleries/reactions for the positive electrode
            when constructing the model.
    """

    def __init__(self, name="Unnamed lumped-parameter lithium-ion cell model"):
        super().__init__(name)

        # Initialize model.
        self._init_model_parameters()
        self._init_model_variables()
        self._init_model_equations()
        self._init_model_events()

        # Collect model output variables.
        self.variables = {
            'time [s]': pybamm.t,
            'time [min]': pybamm.t / 60,
            'time [h]': pybamm.t / 3600,
            'iapp [A]': self.param.iapp,
            'iapp [C-rate]': self.param.iapp / self.param.Q,
            'soc': self.var.soc,
            'soc [%]': self.var.soc * 100,
            'vcell [V]': self.var.vcell,
            'pos_thetas_avg': self.var.pos_thetas_avg,
            'neg_thetas_avg': self.var.neg_thetas_avg,
            'thetae': self.var.thetae,
            'FNe [A]': self.var.FNe,
            'phie [V]': self.var.phie,
            'ie [A]': self.var.ie,
            'pos_thetas': self.var.pos_thetas,
            'neg_thetas': self.var.neg_thetas,
            'thetass': self.var.thetass,
            'pos_thetass': self.var.pos_thetass,
            'neg_thetass': self.var.neg_thetass,
            'Uss [V]': self.var.Uss,
            'phis [V]': self.var.phis,
            'is [A]': self.var.is_,
            'ifdl [A]': self.var.ifdl,
            'pos_if [A]': self.var.pos_if,
            'neg_if [A]': self.var.neg_if,
            'phise [V]': self.var.phise,
            'etas [V]': self.var.etas,
            'neg_phis [V]': self.var.neg_phis,
        }

    @property
    def default_geometry(self):
        return {
            "neg": {self.var.x_eff: {"min": 0, "max": 1}},
            "sep": {self.var.x_eff: {"min": 1, "max": 2}},
            "pos": {self.var.x_pos: {"min": 2, "max": 3}},
            "neg_particle": {self.var.r_neg: {"min": 0, "max": 1}},
            "pos_particle": {self.var.r_pos: {"min": 0, "max": 1}},
        }

    @property
    def default_submesh_types(self):
        return {
            "neg": pybamm.Uniform1DSubMesh,
            "sep": pybamm.Uniform1DSubMesh,
            "pos": pybamm.Uniform1DSubMesh,
            "neg_particle": pybamm.Uniform1DSubMesh,
            "pos_particle": pybamm.Uniform1DSubMesh,
        }

    @property
    def default_spatial_methods(self):
        return {
            "neg": pybamm.FiniteVolume(),
            "sep": pybamm.FiniteVolume(),
            "pos": pybamm.FiniteVolume(),
            "neg_particle": pybamm.FiniteVolume(),
            "pos_particle": pybamm.FiniteVolume(),
        }

    @property
    def default_var_pts(self):
        return {
            self.var.x_neg: 20,
            self.var.x_sep: 10,
            self.var.x_pos: 20,
            self.var.r_neg: 20,
            self.var.r_pos: 20,
        }

    @property
    def default_solver(self):
        return pybamm.CasadiSolver(root_tol=1e-3, atol=1e-6, rtol=1e-6)

    @property
    def default_quick_plot_variables(self):
        return [
            "iapp [C-rate]", "soc [%]", "vcell [V]",
            "thetae", "FNe [A]", ["ie [A]", "is [A]"],
            "ifdl [A]", "phie [V]", "thetass",
        ]

    @property
    def default_parameter_values(self):
        if self.J != 2:
            raise ValueError(f"To use default parameter values, use J=2 MSMR galleries.")
        return pybamm.ParameterValues({
            "iapp [A]": 0.269177,
            "vmin [V]": 3.5,
            "vmax [V]": 4.2,
            "T [K]": 25+273.15,
            "W": 0.237798,
            "psi [V]": 0.001739,
            "Q [Ah]": 0.269177,
            "pos_sigma [Ohm-1]": 241.896,
            "pos_Dsref [s-1]": 0.00083125,
            "pos_kappa [Ohm-1]": 5.76159,
            "pos_taue [s]": 25.4882,
            "pos_theta0": 0.999996,
            "pos_theta100": 0.10836198,
            "pos_Rf [Ohm]": 0.047348,
            "pos_Rdl [Ohm]": 0.023674,
            "pos_Cdl [F]": 0.023674,
            "pos_Uocp0 [V]": 4.2,
            "pos_U0_0 [V]": 4.16756,
            "pos_U0_1 [V]": 4.02477,
            "pos_X_0": 0.39669,
            "pos_X_1": 0.60331,
            "pos_omega_0": 1.12446,
            "pos_omega_1": 1.71031,
            "pos_k0_0 [A]": 0.99738674,
            "pos_k0_1 [A]": 0.71850266,
            "pos_beta_0": 0.5,
            "pos_beta_1": 0.5,
            "eff_kappa [Ohm-1]": 34.78626,
            "eff_taue [s]": 0.644645,
            "neg_k0 [A]": 1.895276,
            "neg_beta": 0.5,
            "neg_Rdl [Ohm]": 0.000156245,
            "neg_Cdl [F]": 0.0012672,
        })

    def _init_model_parameters(self):
        # Applied current input.
        self.param.iapp = pybamm.FunctionParameter("iapp [A]", inputs={"t": pybamm.t})

        # Constants.
        self.param.F = pybamm.Parameter("Faraday constant [C.mol-1]")
        self.param.R = pybamm.Parameter("Ideal gas constant [J.K-1.mol-1]")

        # Cell-wide parameters.
        self.param.kD = pybamm.Parameter("kD")
        self.param.psi = pybamm.Parameter("psi [V]")
        self.param.T = pybamm.Parameter("T [K]")
        self.param.Q = pybamm.Parameter("Q [Ah]")
        self.param.vmin = pybamm.Parameter("vmin [V]")
        self.param.vmax = pybamm.Parameter("vmax [V]")

        # Negative electrode.
        self.param.neg_sigma = pybamm.Parameter("neg_sigma [Ohm-1]")
        self.param.neg_Ds = pybamm.Parameter("neg_Ds [s-1]")
        self.param.neg_kappa = pybamm.Parameter("neg_kappa [Ohm-1]")
        self.param.neg_qe = pybamm.Parameter("neg_qe [s]")
        self.param.neg_theta0 = pybamm.Parameter("neg_theta0")
        self.param.neg_theta100 = pybamm.Parameter("neg_theta100")
        self.param.neg_Rdl = pybamm.Parameter("neg_Rdl [Ohm]")
        self.param.neg_Cdl = pybamm.Parameter("neg_Cdl [F]")
        self.param.neg_Rf = pybamm.Parameter("neg_Rf [Ohm]")
        self.param.neg_k0 = pybamm.Parameter("neg_k0 [A]")
        self.param.neg_alpha = pybamm.Parameter("neg_alpha")
        self.param.neg_thetas0 = pybamm.Parameter("neg_thetas0")
        self.param.neg_Uocp0 = pybamm.Parameter("neg_Uocp0 [V]")  # initial OCP of positive electrode

        # Separator.
        self.param.sep_kappa = pybamm.Parameter("sep_kappa [Ohm-1]")
        self.param.sep_qe = pybamm.Parameter("sep_qe [s]")

        # Positive electrode.
        self.param.pos_sigma = pybamm.Parameter("pos_sigma [Ohm-1]")
        self.param.pos_Ds = pybamm.Parameter("pos_Ds [s-1]")
        self.param.pos_kappa = pybamm.Parameter("pos_kappa [Ohm-1]")
        self.param.pos_qe = pybamm.Parameter("pos_qe [s]")
        self.param.pos_theta0 = pybamm.Parameter("pos_theta0")
        self.param.pos_theta100 = pybamm.Parameter("pos_theta100")
        self.param.pos_Rdl = pybamm.Parameter("pos_Rdl [Ohm]")
        self.param.pos_Cdl = pybamm.Parameter("pos_Cdl [F]")
        self.param.pos_Rf = pybamm.Parameter("pos_Rf [Ohm]")
        self.param.pos_k0 = pybamm.Parameter(f"pos_k0 [A]")
        self.param.pos_alpha = pybamm.Parameter(f"pos_alpha")
        self.param.pos_thetas0 = pybamm.Parameter("pos_thetas0")
        self.param.pos_Uocp0 = pybamm.Parameter("pos_Uocp0 [V]")  # initial OCP of positive electrode

        # NOTE: OCP function parameters appear at end of _init_model_variables().

    def _init_model_variables(self):
        #
        # Spatial variables.
        #

        self.var.r_neg = pybamm.SpatialVariable(
            "r_neg",
            domain=["neg_particle"],
            auxiliary_domains={"secondary": "neg"},
            coord_sys="spherical polar",
        )
        self.var.r_pos = pybamm.SpatialVariable(
            "r_pos",
            domain=["pos_particle"],
            auxiliary_domains={"secondary": "pos"},
            coord_sys="spherical polar",
        )
        self.var.x_neg = pybamm.SpatialVariable("x_neg", domain=["neg"], coord_sys="cartesian")
        self.var.x_sep = pybamm.SpatialVariable("x_sep", domain=["sep"], coord_sys="cartesian")
        self.var.x_pos = pybamm.SpatialVariable("x_pos", domain=["pos"], coord_sys="cartesian")

        #
        # Electrolyte variables.
        #

        # Salt concentration.
        self.var.neg_thetae = pybamm.Variable("neg_thetae", domain="neg")
        self.var.sep_thetae = pybamm.Variable("sep_thetae", domain="sep")
        self.var.pos_thetae = pybamm.Variable("pos_thetae", domain="pos")
        self.var.thetae = pybamm.concatenation(self.var.neg_thetae, self.var.sep_thetae, self.var.pos_thetae)

        # Electrolyte potential.
        self.var.neg_phie = pybamm.Variable("neg_phie [V]", domain="pos")
        self.var.sep_phie = pybamm.Variable("sep_phie [V]", domain="sep")
        self.var.pos_phie = pybamm.Variable("pos_phie [V]", domain="pos")
        self.var.phie = pybamm.concatenation(self.var.neg_phie, self.var.sep_phie, self.var.pos_phie)

        #
        # Solid variables.
        #

        # Solid potential.
        self.var.neg_phis = pybamm.Variable("neg_phis [V]", domain="neg")
        self.var.sep_phis = pybamm.PrimaryBroadcast(pybamm.Scalar(np.nan), "sep")  # phis d.n.e. in sep
        self.var.pos_phis = pybamm.Variable("pos_phis [V]", domain="pos")
        self.var.phis = pybamm.concatenation(self.var.neg_phis, self.var.sep_phis, self.var.pos_phis)

        # Particle stoichiometry.
        self.var.neg_thetas = pybamm.Variable(
            "neg_thetas [V]", domain="neg_particle", auxiliary_domains={"secondary": "neg"})
        self.var.sep_thetas = pybamm.PrimaryBroadcast(np.nan, "sep")  # thetas d.n.e. in sep
        self.var.pos_thetas = pybamm.Variable(
            "pos_thetas [V]", domain="pos_particle", auxiliary_domains={"secondary": "pos"})
        self.var.thetas = pybamm.concatenation(self.var.neg_thetas, self.var.sep_thetas, self.var.pos_thetas)

        # Particle surface stoichiometry.
        self.var.neg_thetass = pybamm.surf(self.var.neg_thetas)
        self.var.sep_thetass = pybamm.PrimaryBroadcast(np.nan, "sep")  # thetass d.n.e. in sep
        self.var.pos_thetass = pybamm.surf(self.var.pos_thetas)
        self.var.thetass = pybamm.concatenation(self.var.neg_thetass, self.var.sep_thetass, self.var.pos_thetass)

        #
        # Interface variables.
        #

        # Faradaic plus double-layer current.
        self.var.neg_ifdl = pybamm.Variable("neg_ifdl [A]", domain="neg")
        self.var.sep_ifdl = pybamm.PrimaryBroadcast(0, "sep")  # ifdl=0 in sep
        self.var.pos_ifdl = pybamm.Variable("pos_ifdl [A]", domain="pos")
        self.var.ifdl = pybamm.concatenation(self.var.neg_ifdl, self.var.sep_ifdl, self.var.pos_ifdl)

        # Double-layer capacitor voltage.
        self.var.neg_vdl = pybamm.Variable("neg_vdl [V]", domain="neg")
        self.var.sep_vdl = pybamm.PrimaryBroadcast(np.nan, "sep")  # vdl d.n.e. in sep
        self.var.pos_vdl = pybamm.Variable("pos_vdl [V]", domain="pos")
        self.var.vdl = pybamm.concatenation(self.var.neg_vdl, self.var.sep_vdl, self.var.pos_vdl)

        #
        # Expressions.
        #

        self.var.f = self.param.F / self.param.R / self.param.T

        # OCP functions.
        self.param.neg_Uss = pybamm.FunctionParameter(
            "neg_Uss [V]", {"neg_thetass": self.var.neg_thetass, "T": self.param.T})
        self.param.pos_Uss = pybamm.FunctionParameter(
            "pos_Uss [V]", {"pos_thetass": self.var.pos_thetass, "T": self.param.T})

    def _init_model_equations(self):

        #
        # Current-overpotential equations.
        #

        # Overpotential.
        self.var.neg_phise = self.var.neg_phis - self.var.neg_phie
        self.var.sep_phise = pybamm.PrimaryBroadcast(np.nan, "sep")
        self.var.pos_phise = self.var.pos_phis - self.var.pos_phie
        self.var.phise = pybamm.concatenation(self.var.neg_phise, self.var.sep_phise, self.var.pos_phise)
        self.var.neg_etas = self.var.neg_phise - self.param.neg_Uss - self.param.neg_Rf * self.var.neg_ifdl
        self.var.sep_etas = pybamm.PrimaryBroadcast(np.nan, "sep")
        self.var.pos_etas = self.var.pos_phise - self.param.pos_Uss - self.param.pos_Rf * self.var.pos_ifdl
        self.var.etas = pybamm.concatenation(self.var.neg_etas, self.var.eff_etas, self.var.pos_etas)

        # Faradaic current.
        self.var.neg_i0 = (
            self.param.neg_k0 *
            self.var.neg_thetass ** self.param.neg_alpha *
            (1 - self.var.neg_thetass) ** (1 - self.param.neg_alpha) *
            self.var.neg_thetae ** (1 - self.param.neg_alpha)
        )
        self.var.neg_if = self.var.neg_i0 * (
            pybamm.exp((1 - self.param.neg_alpha) * self.var.f * self.var.neg_etas)
            - pybamm.exp(-self.param.neg_alpha * self.var.f * self.var.neg_etas)
        )
        self.var.pos_i0 = (
            self.param.pos_k0 *
            self.var.pos_thetass ** self.param.pos_alpha *
            (1 - self.var.pos_thetass) ** (1 - self.param.pos_alpha) *
            self.var.pos_thetae ** (1 - self.param.pos_alpha)
        )
        self.var.pos_if = self.var.pos_i0 * (
            pybamm.exp((1 - self.param.pos_alpha) * self.var.f * self.var.pos_etas)
            - pybamm.exp(-self.param.pos_alpha * self.var.f * self.var.pos_etas)
        )

        # Double-layer current.
        self.var.neg_idl = (self.var.neg_etas + self.var.neg_Uss - self.var.neg_vdl) / self.param.neg_Rdl
        self.var.pos_idl = (self.var.pos_etas + self.var.pos_Uss - self.var.pos_vdl) / self.param.pos_Rdl

        # Current-overpotential equations (neg).
        self.algebraic[self.var.neg_ifdl] = self.var.neg_if + self.var.neg_idl - self.var.neg_ifdl
        self.rhs[self.var.neg_vdl] = (
            (self.var.neg_etas + self.var.neg_Uss - self.var.neg_vdl)
            / self.param.neg_Rdl / self.param.neg_Cdl
        )
        self.initial_conditions[self.var.neg_ifdl] = 0
        self.initial_conditions[self.var.neg_vdl] = self.param.pos_Uocp0

        # Current-overpotential equations (pos).
        self.algebraic[self.var.pos_ifdl] = self.var.pos_if + self.var.pos_idl - self.var.pos_ifdl
        self.rhs[self.var.pos_vdl] = (
            (self.var.pos_etas + self.var.pos_Uss - self.var.pos_vdl)
            / self.param.pos_Rdl / self.param.pos_Cdl
        )
        self.initial_conditions[self.var.pos_ifdl] = 0
        self.initial_conditions[self.var.pos_vdl] = self.param.pos_Uocp0

        #
        # Charge conservation in solid.
        #

        self.var.neg_is = -self.param.neg_sigma * pybamm.grad(self.var.neg_phis)
        self.var.sep_is = pybamm.PrimaryBroadcast(np.nan, "sep")  # is d.n.e. in sep
        self.var.pos_is = -self.param.pos_sigma * pybamm.grad(self.var.pos_phis)
        self.var.is_ = pybamm.concatenation(self.var.neg_is, self.var.sep_is, self.var.pos_is)
        # neg
        self.algebraic[self.var.neg_phis] = pybamm.div(self.var.neg_is) + self.var.neg_ifdl
        self.boundary_conditions[self.var.neg_phis] = {
            "left": (+self.param.iapp / self.param.neg_sigma, "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[self.var.neg_phis] = self.param.neg_Uocp0
        # pos
        self.algebraic[self.var.pos_phis] = pybamm.div(self.var.pos_is) + self.var.pos_ifdl
        self.boundary_conditions[self.var.pos_phis] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-self.param.iapp / self.param.pos_sigma, "Neumann"),
        }
        self.initial_conditions[self.var.pos_phis] = self.param.pos_Uocp0

        #
        # Mass conservation in solid.
        #

        # neg
        self.rhs[self.var.neg_thetas] = pybamm.div(self.param.neg_Ds * pybamm.grad(self.var.neg_thetas))
        self.boundary_conditions[self.var.neg_thetas] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -self.var.neg_if * abs(self.param.neg_theta100 - self.param.neg_theta0)
                / 10_800 / self.param.Q / self.param.neg_Ds,
                "Neumann"
            )
        }
        self.initial_conditions[self.var.neg_thetas] = self.param.neg_thetas0
        # pos
        self.rhs[self.var.pos_thetas] = pybamm.div(self.param.pos_Ds * pybamm.grad(self.var.pos_thetas))
        self.boundary_conditions[self.var.pos_thetas] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -self.var.pos_if * abs(self.param.pos_theta100 - self.param.pos_theta0)
                / 10_800 / self.param.Q / self.param.pos_Ds,
                "Neumann"
            )
        }
        self.initial_conditions[self.var.pos_thetas] = self.param.pos_thetas0

        #
        # Charge conservation in electrolyte.
        #

        self.var.neg_ie = (
            -self.param.neg_kappa * pybamm.grad(self.var.neg_phie)
            + self.param.kD * self.param.neg_kappa * self.param.T * pybamm.grad(self.var.neg_thetae)
        )
        self.var.sep_ie = (
            -self.param.sep_kappa * pybamm.grad(self.var.sep_phie)
            - self.param.kD * self.param.sep_kappa * self.param.T * pybamm.grad(self.var.sep_thetae)
        )
        self.var.pos_ie = (
            -self.param.pos_kappa * pybamm.grad(self.var.pos_phie)
            + self.param.kD * self.param.pos_kappa * self.param.T * pybamm.grad(self.var.pos_thetae)
        )
        self.var.ie = pybamm.concatenation(self.var.neg_ie, self.var.sep_ie, self.var.pos_ie)
        self.algebraic[self.var.neg_phie] = pybamm.div(self.var.neg_ie) - self.var.neg_ifdl
        self.algebraic[self.var.sep_phie] = pybamm.div(self.var.sep_ie) - self.var.sep_ifdl
        self.algebraic[self.var.pos_phie] = pybamm.div(self.var.pos_ie) - self.var.pos_ifdl
        self.boundary_conditions[self.var.neg_phie] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),  # set phi_e=0 at x=0
            "right": (
                -self.param.iapp / self.param.neg_kappa
                - self.param.kD * self.param.T * pybamm.boundary_gradient(self.var.neg_thetae,"right"),
                "Neumann"
            ),
        }
        self.boundary_conditions[self.var.sep_phie] = {
            "left": (pybamm.boundary_value(self.var.neg_phie, "right"), "Dirichlet"),
            "right": (
                -self.param.iapp / self.param.sep_kappa
                + self.param.kD * self.param.T * pybamm.boundary_gradient(self.var.sep_thetae, "right"),
                "Neumann"
            ),
        }
        self.boundary_conditions[self.var.pos_phie] = {
            "left":  (pybamm.boundary_value(self.var.eff_phie, "right"), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann")
        }
        self.initial_conditions[self.var.neg_phie] = 0
        self.initial_conditions[self.var.sep_phie] = 0
        self.initial_conditions[self.var.pos_phie] = 0

        #
        # Mass conservation in electrolyte.
        #

        FNLi_neg = -self.param.neg_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.neg_thetae)
        FNLi_sep = -self.param.sep_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.sep_thetae)
        FNLi_pos = -self.param.pos_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.pos_thetae)
        self.rhs[self.var.neg_thetae] = (
            (self.var.neg_ifdl - pybamm.div(FNLi_neg))
            / 3600 / self.param.neg_qe
        )
        self.rhs[self.var.sep_thetae] = (
            (self.var.sep_ifdl - pybamm.div(FNLi_sep))
            / 3600 / self.param.sep_qe
        )
        self.rhs[self.var.pos_thetae] = (
            (self.var.pos_ifdl - pybamm.div(FNLi_pos))
            / 3600 / self.param.pos_qe
        )
        self.boundary_conditions[self.var.neg_thetae] = {
            # Zero salt flux:
            "left": (pybamm.Scalar(0), "Neumann"),
            # Flux continuity at neg-sep boundary:
            "right": (
                pybamm.boundary_gradient(self.var.sep_thetae, "left")
                * self.param.sep_kappa / self.param.neg_kappa,
                "Neumann"
            ),
        }
        self.boundary_conditions[self.var.sep_thetae] = {
            # Concentration continuity at neg-sep boundary:
            "left": (pybamm.boundary_value(self.var.neg_thetae, "right"), "Dirichlet"),
            # Concentration continuity at sep-pos boundary:
            "right": (pybamm.boundary_value(self.var.pos_thetae, "left"), "Dirichlet"),
        }
        self.boundary_conditions[self.var.pos_thetae] = {
            # Flux continuity at sep-pos boundary:
            "left": (
                pybamm.boundary_gradient(self.var.sep_thetae, "right")
                * self.param.sep_kappa / self.param.pos_kappa,
                "Neumann"
            ),
            # Zero salt flux:
            "right": (pybamm.Scalar(0), "Neumann")
        }
        self.initial_conditions[self.var.neg_thetae] = 1.0
        self.initial_conditions[self.var.sep_thetae] = 1.0
        self.initial_conditions[self.var.pos_thetae] = 1.0

        # Compute salt flux.
        self.var.neg_FNe = -(
            self.param.neg_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.neg_thetae)
            + self.var.neg_ie
        )
        self.var.sep_FNe = -(
            self.param.sep_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.sep_thetae)
            + self.var.sep_ie
        )
        self.var.pos_FNe = -(
            self.param.pos_kappa * self.param.psi * self.param.T * pybamm.grad(self.var.pos_thetae)
            + self.var.pos_ie
        )
        self.var.FNe = pybamm.concatenation(self.var.neg_FNe, self.var.eff_FNe, self.var.pos_FNe)

        #
        # Cell-level quantities.
        #

        # Cell voltage.
        self.var.vcell = (
            pybamm.boundary_value(self.var.pos_phis, "right")
            - pybamm.boundary_value(self.var.neg_phis,"left")
        )

        # Average stoichiometry.
        self.var.neg_thetas_avg = pybamm.Integral(
            pybamm.Integral(
                self.var.neg_thetas,
                self.var.r_neg
            )/(4*np.pi/3),  # average by volume of unit sphere
            self.var.x_neg
        )
        self.var.pos_thetas_avg = pybamm.Integral(
            pybamm.Integral(
                self.var.pos_thetas,
                self.var.r_pos
            )/(4*np.pi/3),  # average by volume of unit sphere
            self.var.x_pos
        )

        # Cell state of charge.
        self.var.soc = (
            (self.var.pos_thetas_avg - self.param.pos_theta0)
            / (self.param.pos_theta100 - self.param.pos_theta0)
        )

    def _init_model_events(self):
        self.events += [
            pybamm.Event("Minimum voltage", self.var.vcell - self.param.vmin),
            pybamm.Event("Maximum voltage", self.param.vmax - self.var.vcell),
        ]
