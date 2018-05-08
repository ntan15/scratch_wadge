Globals2D

N = 4;
mesh = '/Users/jchan985/Desktop/wadges/meshes/periodicSquare2.msh';
[Nv, VX, VY, K, EToV] = MeshReaderGmsh2D(mesh);

StartUp2D;


% plotting nodes
Nplot = 15;
[rp sp] = EquiNodes2D(Nplot); [rp sp] = xytors(rp,sp);
Vp = Vandermonde2D(N,rp,sp)/V;
xp = Vp*x; yp = Vp*y;
% PlotMesh2D; axis on;return

Nq = 2*N+1;
[rq sq wq] = Cubature2D(Nq); % integrate u*v*c
Vq = Vandermonde2D(N,rq,sq)/V;
M = Vq'*diag(wq)*Vq;

Pq = M\(Vq'*diag(wq)); % J's cancel out
xq = Vq*x; yq = Vq*y;
Jq = Vq*J;

[rq1D wq1D] = JacobiGQ(0,0,N);
rfq = [rq1D; -rq1D; -ones(size(rq1D))];
sfq = [-ones(size(rq1D)); rq1D; -rq1D];
wfq = [wq1D; wq1D; wq1D];
Vq1D = Vandermonde1D(N,rq1D)/Vandermonde1D(N,JacobiGL(0,0,N));
% plot(rfq,sfq,'o')
Nfq = length(rq1D);

Vfq = Vandermonde2D(N,rfq,sfq)/V;
Vfqf = kron(eye(3),Vq1D);

% make curvilinear mesh (still unstable?)
Lx = 10;
Ly = 5;
a = 1/2;
x0 = 0; y0 = 0;
% x0 = 0; y0 = 0; Lx = 1; Ly = 1;
xx = (x)/(2*Lx);
yy = (y+5)/(2*Ly);

% vv = sin(pi*xx).*sin(2*pi*yy);
% vv = sin(2*pi*xx).*sin(1*pi*yy);
% color_line3(x,y,vv,vv,'.');return

x = x - 2*a*sin(1*pi*xx).*sin(2*pi*yy);
y = y +   a*sin(2*pi*xx).*sin(1*pi*yy);
% y(abs(y)<1e-8 & abs(x-10)<1e-8) = y(abs(y)<1e-8 & abs(x-10)<1e-8) + 4*a;
% keyboard


xq = Vq*x; yq = Vq*y;
xp = Vp*x; yp = Vp*y;
xf = Vfq*x;    yf = Vfq*y;

if 1
    rp1D = linspace(-1,1,100)';
    Vp1D = Vandermonde1D(N,rp1D)/Vandermonde1D(N,JacobiGL(0,0,N));
    Vfp = kron(eye(Nfaces),Vp1D);
    xfp = Vfp*x(Fmask(:),:);
    yfp = Vfp*y(Fmask(:),:);
    plot(xfp,yfp,'k.')
    hold on
%     plot(x,y,'o')
    axis off
    axis equal
% print(gcf,'-dpng','~/Desktop/wadges/docs/figs/mesh2d_affine_converge3.png')
% print(gcf,'-dpng','~/Desktop/wadges/docs/figs/mesh2d_curved_converge3.png')
%     return
end

%% 3D meshes

Globals3D
N = 4;
mesh = '/Users/jchan985/Desktop/wadges/meshes/periodicCube0.msh';
[Nv, VX, VY, VZ, K, EToV] = MeshReaderGmsh3D(mesh);

StartUp3D

% make curvilinear mesh (still unstable?)
a = 0/2;
x0 = 0; y0 = 0;
xx = x/(10);
yy = y/(20);
zz = z/10;
x = x +   a*sin(pi*xx) .* sin(2*pi*yy).*sin(pi*zz);
y = y - 2*a*sin(2*pi*xx).*sin(pi*yy) .* sin(2*pi*zz);
z = z +   a*sin(pi*xx) .* sin(2*pi*yy).*sin(pi*zz);
xb = []; yb = []; zb = [];
for e = 1:K
    for f = 1:4
        if (EToE(e,f)==e)
            xb = [xb x(Fmask(:,f),e)];
            yb = [yb y(Fmask(:,f),e)];
            zb = [zb z(Fmask(:,f),e)];
        end
    end
end

[rx,sx,tx,ry,sy,ty,rz,sz,tz,J] = GeometricFactors3D(x,y,z,Dr,Ds,Dt);

r1D = linspace(-1,1,100)';
e = ones(size(r1D));
% rf = [r1D; r1D; -e];
% sf = [-e; -r1D; r1D];
% plot(rf,sf,'k.')
% Vt = Vandermonde2D(N,rf,sf)/Vandermonde2D(N,r(Fmask(:,1)),s(Fmask(:,1)));
% plot3(Vt*xb,Vt*yb,Vt*zb,'k.')

rf = [r1D;r1D;-e; -e; r1D;  -e];
sf = [-e; -r1D;r1D;-e; -e;  r1D];
tf = [-e;  -e; -e; r1D;-r1D;-r1D];

Vf = Vandermonde3D(N,rf,sf,tf)/V;
plot3(Vf*x,Vf*y,Vf*z,'k.')

%% TG mesh

Globals3D
N = 4;
mesh = '/Users/jchan985/Desktop/wadges/meshes/periodicCubeTG0.msh';
[Nv, VX, VY, VZ, K, EToV] = MeshReaderGmsh3D(mesh);

StartUp3D

% make curvilinear mesh (still unstable?)
a = 1/2;
x0 = 0; y0 = 0;
xx = x;
yy = y;
zz = z;
x = x +  a*sin(xx) .* sin(yy).*sin(zz);
y = y +  a*sin(xx).*sin(yy) .* sin(zz);
z = z +  a*sin(xx) .* sin(yy).*sin(zz);
xb = []; yb = []; zb = [];
for e = 1:K
    for f = 1:4
        if (EToE(e,f)==e)
            xb = [xb x(Fmask(:,f),e)];
            yb = [yb y(Fmask(:,f),e)];
            zb = [zb z(Fmask(:,f),e)];
        end
    end
end

[rx,sx,tx,ry,sy,ty,rz,sz,tz,J] = GeometricFactors3D(x,y,z,Dr,Ds,Dt);

r1D = linspace(-1,1,100)';
e = ones(size(r1D));
% rf = [r1D; r1D; -e];
% sf = [-e; -r1D; r1D];
% plot(rf,sf,'k.')
% Vt = Vandermonde2D(N,rf,sf)/Vandermonde2D(N,r(Fmask(:,1)),s(Fmask(:,1)));
% plot3(Vt*xb,Vt*yb,Vt*zb,'k.')

rf = [r1D;r1D;-e; -e; r1D;  -e];
sf = [-e; -r1D;r1D;-e; -e;  r1D];
tf = [-e;  -e; -e; r1D;-r1D;-r1D];

Vf = Vandermonde3D(N,rf,sf,tf)/V;
plot3(Vf*x,Vf*y,Vf*z,'k.')

