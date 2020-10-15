function zp = zener_polaron(jvec)
% In this tutorial we will construct the Zener Polaron model for
% Pr(Ca0.1Sr0.9)2Mn2O7 as described in:
% Ground State in a Half-Doped Manganite Distinguished by Neutron Spectroscopy
% G.E. Johnstone, T.G. Perring, O. Sikora, D. Prabhakaran, and A.T. Boothroyd
% Phys. Rev. Lett. 109 237202 (2012), https://arxiv.org/abs/1210.710
%
% The Zener polaron is a strongly coupled pair of Mn atoms which share an
% electron via the Zener double exchange mechanism. They act like a single
% S=7/2 spin, and will be modelled in SpinW as such with the position of
% the Mn_2^7+ polaron being the center of the pair of Mn-Mn atoms.

% Lattice parameters - we simplify to make it tetragonal to decrease the
% number of exchange bonds to be defined later.
lat = [5.5 5.5 20];
alf = [90 90 90];

% Like for the CE model in prcasrmn2o7.m we are forced to use a doubled
% "structural" unit cell to define the model in SpinW (which means that
% hk0 values should be halved to compare between SpinW and the paper).
zp = spinw;
zp.genlattice('lat_const', lat.*[2 2 1], 'angled', alf, 'spgr', 'x,y,-z');
% Note that you will get several warnings that SpinW does not understand
% what a Zener Polaron is and cannot assign it a form factor or scattering
% length - you can ignore this as we will just calculate the dispersion
% with this model.
[~,ffn] = sw_mff('MMn3');
ffx = 'MMn3';
b = sw_nb('MMn3');
zp.addatom('label', 'ZP-up', 'r', [0.125 0.875 0.1], 'S', 7/2, 'color', 'gold', 'formfactn', ffn, 'formfactx', ffx, 'Z', 25, 'b', b);
zp.addatom('label', 'ZP-lf', 'r', [0.625 0.625 0.1], 'S', 7/2, 'color', 'gold', 'formfactn', ffn, 'formfactx', ffx, 'Z', 25, 'b', b);
zp.addatom('label', 'ZP-dn', 'r', [0.125 0.375 0.1], 'S', 7/2, 'color', 'gold', 'formfactn', ffn, 'formfactx', ffx, 'Z', 25, 'b', b);
zp.addatom('label', 'ZP-rg', 'r', [0.625 0.125 0.1], 'S', 7/2, 'color', 'gold', 'formfactn', ffn, 'formfactx', ffx, 'Z', 25, 'b', b);

% Set the magnetic structure
S0 = [0; 1; 0];
S1 = [1; 0; 0];
spin_up = find(~cellfun(@isempty, strfind(zp.table('matom').matom, 'up')));
spin_dn = find(~cellfun(@isempty, strfind(zp.table('matom').matom, 'dn')));
spin_lf = find(~cellfun(@isempty, strfind(zp.table('matom').matom, 'lf')));
spin_rg = find(~cellfun(@isempty, strfind(zp.table('matom').matom, 'rg')));
SS = zeros(3, numel(spin_up)+numel(spin_dn));
SS(:, spin_up) = repmat(S0, 1, numel(spin_up));
SS(:, spin_dn) = repmat(-S0, 1, numel(spin_dn));
SS(:, spin_rg) = repmat(S1, 1, numel(spin_up));
SS(:, spin_lf) = repmat(-S1, 1, numel(spin_dn));
zp.genmagstr('mode', 'direct', 'S', SS)

% Now assign the exchange interactions
% Parameters are scaled up from Ewings et al., Phys. Rev. B 94 014405 (2016)
% https://arxiv.org/abs/1501.01148
if ~exist('jvec', 'var')
    jvec = [[-1.6 -0.11 0.11]*1.5 1.05];
end

JFU = jvec(1);
JFD = jvec(2);
JA = jvec(3);
Jperp = jvec(4);

%zp.gencoupling('forceNoSym', true, 'fid', 0);
zp.addmatrix('label', 'JFU', 'value', JFU, 'color', 'green') % FM between non parallel polarons in +b
zp.addmatrix('label', 'JFD', 'value', JFD, 'color', 'white') % FM between non parallel polarons in -b
zp.addmatrix('label', 'JA', 'value', JA, 'color', 'yellow')  % AFM between parallel polarons
zp.addmatrix('label', 'Jperp', 'value', Jperp, 'color', 'blue')
%plot(zp, 'range', [0 2; 0 2; -0.2 0.2]);

% Print the bond table - look at which bond index corresponds to which J
%zp.table('bond', 1:10)

% Assign the J's to the bonds.
zp.gencoupling('forceNoSym', true, 'fid', 0)
zp.addcoupling('mat', 'Jperp', 'bond', 1)  % dr=[0 0 0.2]
zp.addcoupling('mat', 'JA', 'bond', 2)     % dr=[0 +/-0.5 0]
zp.addcoupling('mat', 'JFU', 'bond', 3, 'atom', {'ZP-up', 'ZP-lf'}) % Within chains
zp.addcoupling('mat', 'JFU', 'bond', 3, 'atom', {'ZP-dn', 'ZP-rg'}) % Within chains
zp.addcoupling('mat', 'JFD', 'bond', 3, 'atom', {'ZP-up', 'ZP-rg'}) % Between chains
zp.addcoupling('mat', 'JFD', 'bond', 3, 'atom', {'ZP-dn', 'ZP-lf'}) % Between chains

%plot(zp, 'range', [0 2; 0 2; -0.2 0.2]);

% Can you see the resemblance between Fig 1c of the Johnstone paper and
% the magnetic structure which has been defined here?

%%
% Optimises the structure
%res = zp.optmagsteep();
%zp = res.obj;
%plot(zp, 'range', [0 2; 0 2; 0 0.2])

% What's happended here?

%%
% Calculates the dispersion

%spec = zp.spinwave({[0 0 0] [2 0 0] [2 0 2] [0 0 2] [0 0 0] 500}, 'hermit', false);
%figure; sw_plotspec(spec);
%specg = sw_egrid(spec, 'Evect', linspace(0,100,2000), 'imagChk', false);
%figure; sw_plotspec(specg,'mode','color','dE',0.5);

% How does this compare to the CE model defined in the file prcasrmn2o7.m?

% Why does this happen? (E.g. what is the defining difference between the
% CE model and the Zener Polaron model as far as the number of spin wave
% modes is concerned?)
