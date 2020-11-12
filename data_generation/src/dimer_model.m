function pcsmo = dimer_model(JFW, JA1, JA2, JFS, Jperp, D)
% The model is based on the paper: 
% Ground State in a Half-Doped Manganite Distinguished by Neutron Spectroscopy
% G.E. Johnstone, T.G. Perring, O. Sikora, D. Prabhakaran, and A.T. Boothroyd
% Phys. Rev. Lett. 109 237202 (2012), https://arxiv.org/abs/1210.7108

%    jvec = [-8.43 1.52 1.52 -14.2 0.92 0.073];
%JFW = str2num(JFW); JA1 = str2num(JA1); JA2 = str2num(JA2); JFS = str2num(JFS); Jperp = str2num(Jperp); D = str2num(D);
disp(sprintf('Parameters: JFW=%f, JA1=%f, JA2=%f, JFS=%f, Jperp=%f, D=%f', JFW, JA1, JA2, JFS, Jperp, D));

% We simplify this model by considering only a single bilayer instead
% the two in the original unit cell.
% However, a complication, is that the CE magnetic structure in this
% material is a 2-k structure so it can only be represented in SpinW
% using the supercell ('direct') method.
%
% The complex geometry of the exchange interactions means that we will use
% different atoms with different labels and use the 'atom' option in addcoupling()
% to specify the bonds exactly.
%
% A problem with the nature of this model, however, in that
% not only is the structure 2-k, but the exchange interactions also do not
% obey the translational symmetry of the high temperature Amam unit cell.
% This means that to express them, we have to use a "structural" unit cell 
% which is the same size as the magnetic unit cell (since SpinW only allows
% you to define atoms within the first unit cell, and forces the bonds to 
% be translationally symmetric in the supercell). This means we have to use
% lattice parameters a and b which are twice as large as in the high 
% temperature Amam structure, which means that the hk0 indices outputted 
% by SpinW should be divided by 2 to get the equivalent positions in the 
% paper.

SM4 = 7/4;   % Spin length for Mn4+
SM3 = 7/4;   % Spin length for Mn3+

pcsmo = spinw();
lat = [5.408 5.4599 19.266]; 
alf = [90 90 90];
pcsmo.genlattice('lat_const', lat.*[2 2 1], 'angled', alf, 'spgr', 'x,y+1/2,-z'); 

% In addition to defining the symmetry, because we are using custom labels
% (e.g. not of the form 'MXXN' where XX is the element abbreviation and N
% is the valence), we have to define the form factor and scattering length
[~,ffn3] = sw_mff('MMn3');
[~,ffn4] = sw_mff('MMn4');
myaddatom3 = @(x,y,z) pcsmo.addatom('r', y, 'S', SM3, 'color', z, ...
    'formfactn', ffn3, 'formfactx', 'MMn3', 'Z', 25, 'b', sw_nb('MMn3'), 'label', x);
myaddatom4 = @(x,y,z) pcsmo.addatom('r', y, 'S', SM4, 'color', z, ...
    'formfactn', ffn4, 'formfactx', 'MMn4', 'Z', 25, 'b', sw_nb('MMn4'), 'label', x);
myaddatom4('Mn4-up', [0 0 0.1], 'gold');
myaddatom4('Mn4-up', [0.5 0.5 0.1], 'gold');
myaddatom4('Mn4-dn', [0 0.5 0.1], 'gold');
myaddatom4('Mn4-dn', [0.5 0 0.1], 'gold');
myaddatom3('Mn3-up', [0.25 0.75 0.1], 'black');
myaddatom3('Mn3-up', [0.75 0.75 0.1], 'black');
myaddatom3('Mn3-dn', [0.25 0.25 0.1], 'black');
myaddatom3('Mn3-dn', [0.75 0.25 0.1], 'black');

% Generate the magnetic structure
S0 = [0; 1; 0];

% Get the list of indices of which are spin up and which are down atoms:
spin_up = find(~cellfun(@isempty, strfind(pcsmo.table('matom').matom, 'up')));
spin_dn = find(~cellfun(@isempty, strfind(pcsmo.table('matom').matom, 'dn')));

SS = zeros(3, 16);
SS(:, spin_up) = repmat(S0, 1, numel(spin_up));
SS(:, spin_dn) = repmat(-S0, 1, numel(spin_dn));
pcsmo.genmagstr('mode', 'direct', 'S', SS)

% Now define the exchange interactions for the Goodenough model.
% We have to force P1 symmetry because the nearest neighbour bonds (intra-
% and inter-chain) do not obey any translational symmetry of the structural
% orthorhombic cell.
pcsmo.gencoupling('forceNoSym', true, 'fid', 0);
% Print out table to determine which bond is which
%pcsmo.table('bond', 1)

%if ~exist('jvec', 'var')
%    jvec = [-8.43 1.52 1.52 -14.2 0.92 0.073];
%end
%JFW = jvec(1); JA1 = jvec(2); JA2 = jvec(3); JFS = jvec(4); Jperp = jvec(5); D = jvec(6);

pcsmo.addmatrix('label', 'JFS', 'value', JFS, 'color', 'green');
pcsmo.addmatrix('label', 'JA1', 'value', JA1, 'color', 'yellow');
pcsmo.addmatrix('label', 'JA2', 'value', JA2, 'color', 'blue');
pcsmo.addmatrix('label', 'JFW', 'value', JFW, 'color', 'red');
pcsmo.addmatrix('label', 'Jperp', 'value', Jperp, 'color', 'white');
pcsmo.addmatrix('label', 'D', 'value', diag([0 0 D]), 'color', 'white');

% The zig-zag chains couple Mn3-Mn4 with same spin.
%pcsmo.addcoupling('mat', 'JFW', 'bond', 1, 'atom', {'Mn3-up', 'Mn4-up'})
%pcsmo.addcoupling('mat', 'JFS', 'bond', 1, 'atom', {'Mn3-dn', 'Mn4-dn'})
% And opposite spins for the inter-chain interaction
%pcsmo.addcoupling('mat', 'JA1', 'bond', 1, 'atom', {'Mn3-up', 'Mn4-dn'})
%pcsmo.addcoupling('mat', 'JA2', 'bond', 1, 'atom', {'Mn3-dn', 'Mn4-up'})
% Second neighbour is the inter-layer coupling with our lattice parameters
pcsmo.addcoupling('mat', 'Jperp', 'bond', 2)
% JF3 couples Mn3 within the same zig-zag (same spin)
%pcsmo.addcoupling('mat', 'JF3', 'bond', 3, 'atom', 'Mn3-up')
%pcsmo.addcoupling('mat', 'JF3', 'bond', 3, 'atom', 'Mn3-dn')
% For the Mn4+ intra-zigzag next nearest neighbour we cannot use just the
% label name, because then we also get an inter-chain coupling (uncomment:
% pcsmo.gencoupling('forceNoSym', true)
% pcsmo.addcoupling('mat', 'JF2', 'bond', 8, 'atom', 'Mn4-up')
% plot(pcsmo, 'range', [0 2; 0 2; 0 0.2])
% And see what it produces).
% We actually only want interactions going in the +b direction which
% originates from the Mn4+ atom which have a=0.5.
% Find indexes of the Mn4+ atoms which have a=0.5:
%idmid = find((~cellfun(@isempty, strfind(pcsmo.table('matom').matom, 'Mn4'))) ...
%    .* (pcsmo.table('matom').pos(:,1)==0.5));
%bond8 = pcsmo.table('bond', 8);
% Finds the bonds which start on one of these atoms and goes along +b
%idstart = find(ismember(bond8.idx1, idmid) .* (bond8.dr(:,2)>0));
% Finds the bonds which ends on one of these atoms and goes along -b
%idend = find(ismember(bond8.idx2, idmid) .* (bond8.dr(:,2)<0));
%pcsmo.addcoupling('mat', 'JF2', 'bond', 8, 'subIdx', [idstart; idend]');
%bond1 = pcsmo.table('bond', 1)

% The zig-zag chains couple Mn3-Mn4 with same spin.
pcsmo.addcoupling('mat', 'JFS', 'bond', 1, 'subIdx', [1 17 21 22]);
pcsmo.addcoupling('mat', 'JFW', 'bond', 1, 'subIdx', [2 15 4 18]);
pcsmo.addcoupling('mat', 'JFS', 'bond', 1, 'subIdx', [31 11 32 26]);
pcsmo.addcoupling('mat', 'JFW', 'bond', 1, 'subIdx', [9 12 27 29]);

% And opposite spins for the inter-chain interaction
pcsmo.addcoupling('mat', 'JA1', 'bond', 1, 'subIdx', [8 10 16 19]);
pcsmo.addcoupling('mat', 'JA2', 'bond', 1, 'subIdx', [13 14 20 23]);
pcsmo.addcoupling('mat', 'JA1', 'bond', 1, 'subIdx', [3 5 24 28]);
pcsmo.addcoupling('mat', 'JA2', 'bond', 1, 'subIdx', [6 7 25 30]);

pcsmo.addaniso('D');

%{
plot(pcsmo, 'range', [0 1; 0 1; 0 0.2])

%%
% Check that the structure we defined is optimum for the give exchanges
res = pcsmo.optmagsteep()
plot(pcsmo, 'range', [0 1; 0 1; 0 0.2])

% How many iterations did optmagsteep take?
% What does this mean?

%%
% Plots the spin wave along some directions

spec = pcsmo.spinwave({[0 0 0] [2 0 0] [2 0 2] [0 0 2] [0 0 0] 500}, 'hermit', false);
figure; sw_plotspec(spec);
specg = sw_egrid(spec, 'Evect', linspace(0,100,2000));
figure; sw_plotspec(specg, 'mode', 'color', 'dE',0.5);
%}

% Does the dispersion agree with the paper?

%%
%

% Define twins
pcsmo.twin.rotc(:,:,2) = [0 1 0; 1 0 0; 0 0 0];
pcsmo.twin.vol = [0.5 0.5];

