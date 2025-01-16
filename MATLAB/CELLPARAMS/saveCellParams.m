% saveCellParams.m
%
% Fetch cell parameters and save to .mat file.
%
% -- Changelog --
% 2024.09.23 | Created | Wesley Hileman <whileman@uccs.edu>

addpath('..');
TB.addpaths;

% Constants.
cellFile = 'cellLMO-P2DM.xlsx';  % Name of cell parameters spreadsheet.
TdegC = 25;

% Load cell model from spreadsheet.
p2dm = loadCellModel(cellFile);  % pseudo two-dimensional DFN model

% Convert standard cell model to lumped-parameter Warburg-resistance model
% with dll and sep combined into eff layer.
lpm = convertCellModel(p2dm,'RLWRM');

% Fetch cell capacity.
QAh = getCellParams(lpm,'const.Q');

% Fetch cell parameters.
cellParams = getCellParams(lpm,'TdegC',TdegC);
dataDs = MSMR(cellParams.pos).Ds(cellParams.pos);
DsAvg = 10.^mean(log10(dataDs.Ds));
cellParams.pos.DsAvg = DsAvg;

save('cellParams.mat','cellParams','dataDs');