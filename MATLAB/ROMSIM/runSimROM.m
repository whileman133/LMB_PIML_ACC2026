% runSimROM.m
%
% Simulate ROM.
%
% -- Changelog --
% 2025.01.15 | Created | Wesley Hileman <whileman@uccs.edu>

clear; close all; clc;
addpath(fullfile('..','..'));
TB.addpaths;

romFile = 'cellLMO-P2DM_defaultHRA';
romData = load(fullfile('ROM_FILES',[romFile '.mat']));
ROM = romData.ROM;
LLPM = romData.LLPM;
Rc = getCellParams(LLPM,'const.Rc');
Rfn = getCellParams(LLPM,'neg.Rf');

files = dir(fullfile('.', 'iapp_profiles', '*.xlsx'));
for file = files'
    % Fetch simulation waveform.
    tab = readtable(fullfile(file.folder,file.name));
    time = tab.time;
    iapp = tab.iapp;
    socPct0 = tab.soc0Pct(1);
    TdegC = tab.TdegC(1);
    Tvect = TdegC*ones(size(time));

    % Build simulation struct.
    simData = struct;
    simData.SOC0 = socPct0;
    simData.Iapp = iapp;
    simData.T = Tvect;
    simData.time = time;
    simData.TSHIFT = 0;

    % Run simulation.
    ROMout = simROM(ROM,simData,'outBlend');
    % PyBaMM model does not include series resistances!
    ROMout.Vcell = ROMout.Vcell + (Rc+Rfn)*ROMout.Iapp;

    % Save results.
    [~,fname] = fileparts(file.name);
    save(fullfile('SIM_FILES',[fname '.mat']),'ROMout','simData');
end
