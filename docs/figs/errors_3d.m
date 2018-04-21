% 2D results

err = [1.067171 0.1496387 0.02569296 0.003428271]; % N = 2
errc = [1.377095 0.2195012 0.03858958 0.003990601]; % N = 2, a = 1/2

err = [0.5078199 0.05160533 0.002494255 0.0001106185]; % N = 3
errc = [0.5350362 0.06574176 0.005015359 0.0002430048]; % N = 3, a = 1/2
% 
err = [0.2353737 0.01257141 0.0003215585 1.06097e-05]; % N = 4
errc = [0.297758  0.02076038 0.000816006 2.361021e-05]; % N = 4 curved, a = 1/2, CFL = .25

h = 2*.5.^(0:length(err)-1);
loglog(h,err,'o--')
hold on;
h = 2*.5.^(0:length(err)-1);
loglog(h,errc,'x--')
% loglog(h,.01*h.^4.5,'k--')

print_pgf_coordinates(h,err)
print_pgf_coordinates(h,errc)

%% 3D results

err   = [1.055192  0.1435153 0.02126818 ]; % N = 2
errc  = [1.20915   0.2068999 0.02845052 ]; % N = 2
errck = [1.209334  0.2069238 0.02845067]; % N = 2, kopriva geofacs

err = [ 0.3393179 0.03143425 0.001976993]; % N = 3
errc  = [0.4646133 0.05133686 0.00334595]; % N = 3
errck = [0.4646378 0.05133794 0.003346182]; % N = 3, kopriva geofacs

% err = [ 0.1222291  0.004884339 0.0001924525]; % N = 4
% errc = [0.1845467  0.01043607  0.0003960681]; % N = 4
% errck = [0.1845734 0.01043634]; % N = 4, kopriva geofacs

h = 2*.5.^(0:length(err)-1);
loglog(h,err,'o--')
hold on;
loglog(h,errc,'o--')
loglog(h,errck,'x--')
loglog(h,.005*h.^5,'k--')



print_pgf_coordinates(h,err)
print_pgf_coordinates(h(1:2),errc)
