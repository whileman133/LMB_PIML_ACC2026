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
ts = ROM.xraData.Tsamp;

PP = zeros(ROM.xraData.nfinal+1,length(ROM.ROMmdls));
TAU = zeros(ROM.xraData.nfinal,length(ROM.ROMmdls));
TAU3 = zeros(3,length(ROM.ROMmdls));
for idxROM = 1:length(ROM.ROMmdls)
  rom = ROM.ROMmdls(idxROM);
  pp = eig(rom.A);
  tau = -ts./log(pp(pp<1));
  tau3 = unique(tau);
  tau3 = tau3(end-2:end);
  PP(:,idxROM) = pp;
  TAU(:,idxROM) = tau;
  TAU3(:,idxROM) = tau3;
end % for

socPct = ROM.xraData.SOC;
idx = 10<=socPct&socPct<=90;
TAU3m = TAU3(:,idx);
tau3 = mean(TAU3m,2);
tau3

