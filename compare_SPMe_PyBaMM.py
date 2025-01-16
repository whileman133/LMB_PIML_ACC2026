import os
from dataclasses import dataclass

import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmcrameri.cm as cmc

import util
from spme import SPMe

plt.style.use(['tableau-colorblind10', './thesisformat-lg.mplstyle'])


if __name__ == '__main__':
    cell = util.load_mat_cell_model(os.path.join('MATLAB', 'CELLPARAMS', 'cellParams.mat'))
    # simData = util.load_mat_comsol_sim(os.path.join('matlab_data', 'cc-100dC-100-50.mat'))

    # Define constants.
    ts = 1          # sampling interval [s]
    TdegC = 25      # cell temperature [degC]
    C_rate = 1.0
    soc0_pct = 60
    socf_pct = 50
    t_rest = 1000
    xpos = np.linspace(1, 2, 20)

    # solidSim = SolidFVS(cell, N=10, ts=ts)
    # electrolyteSim = ElectrolyteFVS(cell, Ne=5, Np=10, ts=ts)
    spme = SPMe(cell, ns=10, ne_eff=5, ne_pos=10, ts=ts, TdegC=TdegC)
    r = spme.solid_fvs.edges
    xThetaeSPM = spme.electrolyte_fvs.edges

    # Run PyBaMM simulation.
    sim = util.sim_cc(cell, cell.cst.QAh*C_rate, soc0_pct, socf_pct, TdegC, ts, t_rest)
    time = sim.solution['time [s]']()
    iapp = sim.solution['iapp [A]']()
    tend = np.argwhere(iapp==0)[0][0]
    thetass2 = sim.solution['thetass'](x=2)
    thetass = sim.solution['pos_thetass'](x=xpos)
    thetas2 = sim.solution['pos_thetas'](x=2, r=r)
    vcell_pybamm = sim.solution['vcell [V]']()

    # Run SPMe simulation.
    soln = spme.run(iapp, soc0_pct=soc0_pct)
    ThetassSPM = spme.solid_fvs.thetas(Xs=soln.Xs, r=1)
    ThetasSPM = soln.Xs
    ThetaeSPM = soln.Xe
    vcell_spme = soln.vcell
    rmseThetass = np.sqrt(np.mean((ThetassSPM - thetass2)**2))
    print('RMSE(Thetass):', rmseThetass)

    # Plot vcell
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(time, vcell_pybamm, label='PyBaMM')
    plt.plot(time, vcell_spme, label='SPMe')
    plt.xlabel(r'Time, $t$ [sec]')
    plt.ylabel(r'Cell Voltage, $v_\mathrm{cell}(t)$')
    plt.title(r'Cell Voltage')
    plt.legend()
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'vcell.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'vcell.eps'), bbox_inches='tight')

    # Plot Thetass
    # Vs time:
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1/scipy.constants.golden)
    plt.plot(time, thetass2, color='C0', label='Newman')
    plt.plot(time, ThetassSPM, color='C2', label='SPMe')
    plt.xlabel(r'Time, $t$ [sec]')
    plt.ylabel(r'$\theta_\mathrm{ss}(\tilde{x}=2)$')
    plt.title(r'Surf. Stoich. at $\tilde{x}=2$ vs. Time')
    plt.legend()
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'thetass2.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'thetass2.eps'), bbox_inches='tight')
    # Vs electrode thickness:
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(xpos, thetass[:,tend], color='C0', label='Newman')
    plt.plot(xpos, np.ones_like(xpos)*ThetassSPM[tend], color='C2', label='SPMe')
    plt.xlabel(r'Linear Position, $\tilde{x}$')
    plt.ylabel(r'$\theta_\mathrm{ss}(\tilde{x}=2)$')
    plt.title(r'Surf. Stoich. at $t=380\,\mathrm{s}$ vs. Position')
    plt.legend()
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'thetass-tend.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'thetass-tend.eps'), bbox_inches='tight')
    exit()

    # Plot if
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(time, sim.solution['pos_if [A]'](x=2), label='PyBaMM')
    plt.plot(time, -soln.iapp, label='SPMe')
    plt.xlabel(r'Time, $t$ [sec]')
    plt.ylabel(r'$i_\mathrm{f}(\tilde{x}=2)$ [A]')
    plt.title(r'Faradaic Current')
    plt.legend()
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'if.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'if.eps'), bbox_inches='tight')

    # Plot phie
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(time, sim.solution['phie [V]'](x=2), 'r', label='PyBaMM')
    plt.plot(time, soln.phie2, 'k', label='SPMe')
    plt.xlabel(r'Time, $t$ [sec]')
    plt.ylabel(r'$\phi_\mathrm{e}(\tilde{x}=2)$ [V]')
    plt.title(r'Electrolyte Potential')
    plt.legend()
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'phie.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'compare_SPMe_PyBaMM', 'phie.eps'), bbox_inches='tight')

    plt.show()
    exit()

    # Plot Thetass across electrode versus time.
    fig3, ax3 = plt.subplots(1, 3, figsize=(15, 3.5))
    ax3[0].set_box_aspect(1/scipy.constants.golden)
    ax3[1].set_box_aspect(1/scipy.constants.golden)
    ax3[2].set_box_aspect(1/scipy.constants.golden)
    ax3[2].plot(time, iapp / cell.cst.QAh)
    line_iapp, = ax3[2].plot(time[0], iapp[0] / cell.cst.QAh, 'ro')
    ax3[2].set_title("Applied Current")
    ax3[2].set_xlabel(r"Time, $t$ [sec]")
    ax3[2].set_ylabel(r"$i_\mathrm{app}$ [C-rate]")
    line_thetas, = ax3[0].plot(xpos, thetass[:, 0], 'r', label='PyBaMM')
    line_spm, = ax3[0].plot(xpos, ThetassSPM[0] * np.ones_like(xpos), 'k', label='SPMe')
    ax3[0].set_xlim(min(xpos), max(xpos))
    ax3[0].set_ylim(np.min(thetass), np.max(thetass))
    ax3[0].set_title(r"Surface Stoichiometry vs. Linear Position")
    ax3[0].set_xlabel(r"Linear Position, $\tilde{x}$")
    ax3[0].set_ylabel(r"Surface Stoichiometry, $\theta_\mathrm{ss}$")
    ax3[0].legend()
    line_thetas_r, = ax3[1].plot(r, thetas2[:, 0], color='r', label='PyBaMM')
    line_spm_r = ax3[1].stairs(ThetasSPM[0, :], r, baseline=None, color='k', label='SPMe')
    ax3[1].set_xlim(min(r), max(r))
    ax3[1].set_ylim(np.min(ThetasSPM), np.max(ThetasSPM))
    ax3[1].set_title(r"Stoichiometry vs. Radial Position at $\tilde{x}=2$")
    ax3[1].set_xlabel(r"Radial Position, $\tilde{r}$")
    ax3[1].set_ylabel(r"Stoichiometry, $\theta_\mathrm{s}$")
    ax3[1].legend()
    fig3.tight_layout()
    # Create animation
    div = 10
    def update_Thetass(frame):
        f = div*frame
        line_iapp.set_data([time[f]], [iapp[f] / cell.cst.QAh])
        line_thetas.set_data(xpos, thetass[:, f])
        line_spm.set_data(xpos, ThetassSPM[f] * np.ones_like(xpos))
        line_thetas_r.set_data(r, thetas2[:, f])
        line_spm_r.set_data(ThetasSPM[f, :], r)
        return line_iapp, line_thetas, line_spm, line_thetas_r, line_spm_r
    ani = animation.FuncAnimation(
        fig3, update_Thetass,
        frames=thetass.shape[1]//div,
        interval=ts, blit=False, repeat=False)
    ani.save(os.path.join('plots', 'compare_SPMe_PyBaMM', 'thetas.gif'), writer='imagemagick', fps=10)
    plt.show()

    # Plot Thetae across electrode versus time.
    fig4, ax4 = plt.subplots(2, 1, figsize=(6, 7))
    ax4[1].plot(simData.time, simData.iapp / cell.cst.QAh)
    line_iapp2, = ax4[1].plot(simData.time[0], simData.iapp[0] / cell.cst.QAh, 'ro')
    ax4[1].set_title("Applied Current")
    ax4[1].set_xlabel(r"Time, $t$ [sec]")
    ax4[1].set_ylabel(r"$i_\mathrm{app}$ [C-rate]")
    line_thetae, = ax4[0].plot(simData.xthetae, simData.thetae[0, :], label='COMSOL')
    line_spm2 = ax4[0].stairs(ThetaeSPM[0, :], xThetaeSPM, baseline=None, label='SPM')
    ax4[0].legend()
    ax4[0].set_xlim(min(simData.xthetae), max(simData.xthetae))
    ax4[0].set_ylim(np.min(simData.thetae), np.max(simData.thetae))
    ax4[0].set_title("Electrolyte Composition")
    ax4[0].set_xlabel(r"Linear Position, $\tilde{x}$")
    ax4[0].set_ylabel(r"Li Composition, $\theta_\mathrm{e}=c_\mathrm{e}/c_\mathrm{e,0}$")
    plt.tight_layout()
    # Create animation
    def update_Thetae(frame):
        line_iapp2.set_data([simData.time[frame]], [simData.iapp[frame] / cell.cst.QAh])
        line_thetae.set_data(simData.xthetae, simData.thetae[frame, :])
        line_spm2.set_data(ThetaeSPM[frame, :], xThetaeSPM)
        return (line_iapp2, line_thetae, line_spm2)
    ani2 = animation.FuncAnimation(
        fig4, update_Thetae,
        frames=simData.thetae.shape[0],
        interval=simData.ts/100, blit=False, repeat=False)
    plt.show()

