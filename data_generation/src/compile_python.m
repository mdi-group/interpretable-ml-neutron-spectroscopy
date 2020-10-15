% checks the paths have been set
if ~exist('sw_rootdir')
	addpath(genpath('spinw'));
end

% compile the code
swr = fileparts(sw_rootdir);
mccCommand = ['mcc -W python:spinw '...
    '-d ' swr '/standalone '...
    '-a ' swr '/dat_files/* '...
    '-a ' swr '/external '...
    '-a ' swr '/swfiles '...
    '-v '...
    '-a ' swr '/ass.m '...
    '-a ' swr '/ev.m '...
    '-a ' swr '/evo.m '...
    '-a ' swr '/prcasrmn2o7.m '...
    '-a ' swr '/dimer.m '...
    '-a ' swr '/zener_polaron.m '...
    swr '/sw_main.m'];

eval(mccCommand);
