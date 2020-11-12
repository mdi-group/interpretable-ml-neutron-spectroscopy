function sw_main(varargin)
% the main application of the deployed SpinW app
fprintf('!==================================================================!\n')
fprintf('!                      SPINW                                       !\n')
fprintf('!------------------------------------------------------------------!\n')
fprintf('! Sandor Toth: 2010-2016                                           !\n')
try
    [V,E]=eig_omp(rand(5,5,2));
    [R,P]=chol_omp(rand(5,5,2));
    M = sw_mtimesx(rand(5,5,2),rand(5,5,2));
    fprintf('! Mex code:    available - set ''useMex'' option to use mex files.   !\n')
catch
    fprintf('! Mex code:    Disabled  or not supported on this platform         !\n')
end
fprintf('!------------------------------------------------------------------!\n')
end

