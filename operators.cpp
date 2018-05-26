#include "operators.h"
#include "node_data.h"
#include <cmath>
#include <cstdint>
#include <iostream>
using namespace std;

// ================== test eigen ================

void test_eigen()
{
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);

  cout << "in operators: " << m << endl;
}

void test_solve()
{
  MatrixXd A(3, 3);
  A(0, 0) = 1.0;
  A(0, 1) = 1.0;
  A(0, 2) = 0.0;
  A(1, 0) = 0.0;
  A(1, 1) = 2.0;
  A(1, 2) = 2.0;
  A(2, 0) = 0.0;
  A(2, 1) = 0.0;
  A(2, 2) = 3.0;
  MatrixXd b(3, 1);
  b(0, 0) = 1.0;
  b(1, 0) = 2.0;
  b(2, 0) = 3.0;

  MatrixXd x = mldivide(A, b);
  cout << "matrix = " << A << endl;
  cout << "rhs = " << b << endl;
  cout << "solution = " << x << endl;
  MatrixXd err = (A * x - b);
  double relative_error = err.norm() / b.norm(); // norm() is L2 norm
  printf("Err = %f, rel err = %f\n", err.norm(), relative_error);

  printf("mrdivide test:\n");
  MatrixXd bt = b.transpose();
  x = mrdivide(bt, A);
  cout << "matrix = " << A << endl;
  cout << "rhs = " << b << endl;
  cout << "solution = " << x << endl;
  // cout << "Err is " << err << ", while relative error is: " << relative_error
  // << endl;
}

void test_basis()
{
  int N = 2;
  VectorXd r(N + 1);
  for (int i = 0; i < N + 1; ++i)
  {
    r(i) = -1.0 + 2.0 * i / N;
  }

  VectorXd P = JacobiP(r, 0.0, 0.0, N);
  VectorXd dP = GradJacobiP(r, 0.0, 0.0, N);
  //  cout << "P = " << P << ", gradP = " << dP << endl;

  MatrixXd V2 = Vandermonde2D(N, r, r);
  cout << "V2 = " << V2 << endl;

  VectorXd dP3 = Simplex3DP(r, r, r, 0, 1, 2);
  // cout << "dP3 = " << dP3 << endl;

  VectorXd pr, ps, pt;
  GradSimplex3DP(r, r, r, 1, 1, 1, pr, ps, pt);
  cout << "pr = " << pr << ", ps = " << ps << ", pt = " << pt << endl;

  VectorXd a, b, c;
  rsttoabc(r, r, r, a, b, c);
  // cout << "a = " << a << ", b = " << b << ", c = " << c << endl;

  MatrixXd V = Vandermonde3D(N, r, r, r);
  MatrixXd Vr, Vs, Vt;
  GradVandermonde3D(N, r, r, r, Vr, Vs, Vt);
  //  cout << "V = " << V << endl;
  //  cout << "Vr = " << Vr << endl;
  //  cout << "Vs = " << Vs << endl;
  //  cout << "Vt = " << Vt << endl;

  MatrixXd VBtri = BernTri(N, r, r);
  MatrixXd VBtet = BernTet(N, r, r, r);
  // cout << "VBtri = " << endl << VBtri << endl;
  // cout << "VBtet = " << endl << VBtet << endl;

  MatrixXd VB1, VB2, VB3, VB4;
  GradBernTet(N, r, r, r, VB1, VB2, VB3, VB4);
  cout << "VB1 = " << endl << VB1 << endl;
  cout << "VB2 = " << endl << VB2 << endl;
  cout << "VB3 = " << endl << VB3 << endl;
  cout << "VB4 = " << endl << VB4 << endl;
}

ref_elem_data *build_ref_ops_2D(int N, int Nq, int Nfq)
{

  // ======= 2D case

  VectorXd r, s;
  Nodes2D(N, r, s);

  VectorXd rq, sq, wq;
  tri_cubature(Nq, rq, sq, wq);

  // cout << "rq = " << rq << endl;

  // nodal
  MatrixXd V = Vandermonde2D(N, r, s);
  MatrixXd Vr, Vs;
  GradVandermonde2D(N, r, s, Vr, Vs);
  MatrixXd Dr = mrdivide(Vr, V);
  MatrixXd Ds = mrdivide(Vs, V);

  // quadrature
  MatrixXd Vqtmp = Vandermonde2D(N, rq, sq);
  MatrixXd Vq = mrdivide(Vqtmp, V);
  MatrixXd M = Vq.transpose() * wq.asDiagonal() * Vq;
  MatrixXd VqW = Vq.transpose() * wq.asDiagonal();
  MatrixXd Pq = mldivide(M, VqW);

  // face quadrature
  VectorXd r1D = r.head(N + 1);
  MatrixXd V1D = Vandermonde1D(N, r1D);

  int Nfaces = 3;
  VectorXd rq1D, wq1D;
  //  cout << "Nq = " << Nq << ", Nfq = " << Nfq << endl;
  JacobiGQ((int)ceil(Nfq / 2.0), 0, 0, rq1D, wq1D);
  MatrixXd rrfq(rq1D.rows(), Nfaces), ssfq(rq1D.rows(), Nfaces);
  VectorXd ones = MatrixXd::Ones(rq1D.rows(), 1);
  rrfq.col(0) = rq1D;
  ssfq.col(0) = -ones;
  rrfq.col(1) = -rq1D;
  ssfq.col(1) = rq1D;
  rrfq.col(2) = -ones;
  ssfq.col(2) = -rq1D;
  VectorXd rfq = flatten(rrfq);
  VectorXd sfq = flatten(ssfq);

  // reference normals
  MatrixXd nnrJ(rq1D.rows(), Nfaces), nnsJ(rq1D.rows(), 3);
  nnrJ.col(0).fill(0.0);
  nnsJ.col(0).fill(-1.0);

  nnrJ.col(1).fill(1.0);
  nnsJ.col(1).fill(1.0);

  nnrJ.col(2).fill(-1.0);
  nnsJ.col(2).fill(0.0);

  VectorXd nrJ = flatten(nnrJ);
  VectorXd nsJ = flatten(nnsJ);

  //  VectorXd rfq(Map<VectorXd>(rrfq.data(), rrfq.cols()*rrfq.rows()));
  //  VectorXd sfq(Map<VectorXd>(ssfq.data(), ssfq.cols()*ssfq.rows()));

  VectorXd wfq = wq1D.replicate(3, 1);
  MatrixXd Vfqtmp = Vandermonde1D(N, rq1D);
  Vfqtmp = mrdivide(Vfqtmp, V1D);
  MatrixXd Imat = MatrixXd::Identity(3, 3); // face identity matrix
  MatrixXd Vfqf = kron(Imat, Vfqtmp);

  Vfqtmp = Vandermonde2D(N, rfq, sfq);
  MatrixXd Vfq = mrdivide(Vfqtmp, V);
  MatrixXd Mfq = Vfq.transpose() * wfq.asDiagonal();
  MatrixXd Lq = mldivide(M, Mfq);
  MatrixXd VqLq = Vq * Lq;

  MatrixXd Drq = Vq * Dr * Pq - .5 * Vq * Lq * nrJ.asDiagonal() * Vfq * Pq;
  MatrixXd Dsq = Vq * Ds * Pq - .5 * Vq * Lq * nsJ.asDiagonal() * Vfq * Pq;

  ref_elem_data *ref_data = new ref_elem_data;

  ref_data->N = N;
  ref_data->Nq = Nq;
  ref_data->Nfaces = 3;
  ref_data->r = r;
  ref_data->s = s;
  ref_data->V = V;
  ref_data->Dr = Dr;
  ref_data->Ds = Ds;

  ref_data->rq = rq;
  ref_data->sq = sq;
  ref_data->wq = wq;
  ref_data->Vq = Vq;
  ref_data->Pq = Pq;

  ref_data->rfq = rfq;
  ref_data->sfq = sfq;
  ref_data->wfq = wfq;

  ref_data->ref_rfq = rq1D;

  ref_data->Vfqf = Vfqf;
  ref_data->Vfq = Vfq;

  ref_data->nrJ = nrJ;
  ref_data->nsJ = nsJ;

  return ref_data;
}

ref_elem_data *build_ref_ops_3D(int N, int Nq, int Nfq)
{
  ref_elem_data *ref_data = new ref_elem_data;
  // ======= 3D case

  VectorXd r, s, t;
  Nodes3D(N, r, s, t);

  VectorXd rq, sq, tq, wq;
  tet_cubature(Nq, rq, sq, tq, wq);

  // nodal
  MatrixXd V = Vandermonde3D(N, r, s, t);
  MatrixXd Vr, Vs, Vt;
  GradVandermonde3D(N, r, s, t, Vr, Vs, Vt);
  MatrixXd Dr = mrdivide(Vr, V);
  MatrixXd Ds = mrdivide(Vs, V);
  MatrixXd Dt = mrdivide(Vt, V);

  // quadrature
  MatrixXd Vqtmp = Vandermonde3D(N, rq, sq, tq);
  MatrixXd Vq = mrdivide(Vqtmp, V);

  MatrixXd MM = Vq.transpose() * wq.asDiagonal() * Vq;
  MatrixXd VqW = Vq.transpose() * wq.asDiagonal();
  MatrixXd Pq = mldivide(MM, VqW);

  // face quadrature
  int Nfp = (N + 1) * (N + 2) / 2;
  VectorXd rtri = r.head(Nfp);
  VectorXd stri = s.head(Nfp);
  MatrixXd Vtri = Vandermonde2D(N, rtri, stri);

  int Nfaces = 4;
  VectorXd rqtri, sqtri, wqtri;
  //  cout << "Nq = " << Nq << ", Nfq = " << Nfq << endl;
  tri_cubature(Nfq, rqtri, sqtri, wqtri);

  MatrixXd rrfq(rqtri.rows(), Nfaces), ssfq(rqtri.rows(), Nfaces),
      ttfq(rqtri.rows(), Nfaces);
  VectorXd ones = MatrixXd::Ones(rqtri.rows(), 1);
  rrfq.col(0) = rqtri;
  ssfq.col(0) = sqtri;
  ttfq.col(0).fill(-1.0);

  rrfq.col(1) = rqtri;
  ssfq.col(1).fill(-1.0);
  ttfq.col(1) = sqtri;

  rrfq.col(2) = -(ones + rqtri + sqtri);
  ssfq.col(2) = rqtri;
  ttfq.col(2) = sqtri;

  rrfq.col(3).fill(-1.0);
  ssfq.col(3) = rqtri;
  ttfq.col(3) = sqtri;

  VectorXd rfq = flatten(rrfq);
  VectorXd sfq = flatten(ssfq);
  VectorXd tfq = flatten(ttfq);

  // reference normals
  MatrixXd nnrJ(rqtri.rows(), Nfaces), nnsJ(rqtri.rows(), Nfaces),
      nntJ(rqtri.rows(), Nfaces);
  nnrJ.col(0).fill(0.0);
  nnsJ.col(0).fill(0.0);
  nntJ.col(0).fill(-1.0);

  nnrJ.col(1).fill(0.0);
  nnsJ.col(1).fill(-1.0);
  nntJ.col(1).fill(0.0);

  nnrJ.col(2).fill(1.0);
  nnsJ.col(2).fill(1.0);
  nntJ.col(2).fill(1.0);

  nnrJ.col(3).fill(-1.0);
  nnsJ.col(3).fill(0.0);
  nntJ.col(3).fill(0.0);

  VectorXd nrJ = flatten(nnrJ);
  VectorXd nsJ = flatten(nnsJ);
  VectorXd ntJ = flatten(nntJ);

  VectorXd wfq = wqtri.replicate(Nfaces, 1);
  MatrixXd Vfqtmp = Vandermonde2D(N, rqtri, sqtri);
  Vfqtmp = mrdivide(Vfqtmp, Vtri);
  MatrixXd Imat = MatrixXd::Identity(Nfaces, Nfaces); // face identity matrix
  MatrixXd Vfqf = kron(Imat, Vfqtmp);

  Vfqtmp = Vandermonde3D(N, rfq, sfq, tfq);
  MatrixXd Vfq = mrdivide(Vfqtmp, V);

  ref_data->N = N;
  ref_data->Nq = Nq;
  ref_data->Nfaces = 4;
  ref_data->r = r;
  ref_data->s = s;
  ref_data->t = t;
  ref_data->V = V;
  ref_data->Dr = Dr;
  ref_data->Ds = Ds;
  ref_data->Dt = Dt;

  ref_data->rq = rq;
  ref_data->sq = sq;
  ref_data->tq = tq;
  ref_data->wq = wq;
  ref_data->Vq = Vq; // interp to quad pts
  ref_data->Pq = Pq; // project from quad pts

  ref_data->rfq = rfq;
  ref_data->sfq = sfq;
  ref_data->tfq = tfq;
  ref_data->wfq = wfq;

  ref_data->ref_rfq = rqtri;
  ref_data->ref_sfq = sqtri;

  ref_data->Vfqf = Vfqf; // trace dim(d-1) interpolation matrix
  ref_data->Vfq = Vfq;   // interp to face surface nodes

  ref_data->nrJ = nrJ; // reference element normals
  ref_data->nsJ = nsJ;
  ref_data->ntJ = ntJ;

  return ref_data;
}

geo_elem_data *build_geofacs_2D(ref_elem_data *ref_data,
                                const Ref<MatrixXd> EToVX)
{
  VectorXd r = ref_data->r;
  VectorXd s = ref_data->s;

#if 0
  int N = ref_data->N;
  double a = .05;
  VectorXd dr = Eigen::pow(r.array() + s.array(), N);
  VectorXd ds = Eigen::pow(s.array() + r.array(), N);
  VectorXd x = r - a / 1.0 * dr;
  VectorXd y = s + a / 2.0 * ds;
#else
  // TODO double check the order here
  // a b c, a c b, c a b,
  MatrixXd vxa = EToVX.row(0);
  MatrixXd vya = EToVX.row(1);

  MatrixXd vxb = EToVX.row(2);
  MatrixXd vyb = EToVX.row(3);

  MatrixXd vxc = EToVX.row(4);
  MatrixXd vyc = EToVX.row(5);

  VectorXd one = VectorXd::Ones(r.size());
  MatrixXd x = 0.5 * (-(r + s) * vxa + (one + r) * vxb + (one + s) * vxc);
  MatrixXd y = 0.5 * (-(r + s) * vya + (one + r) * vyb + (one + s) * vyc);
#endif


#if 1
  // add curvilinear perturbation to [0,20] x [-5,5]
  MatrixXd xx = x.array()/20.0;
  MatrixXd yy = (y.array()+5.0)/10.0;

  double a = .5;
  x.array() += 2.0*a*(M_PI*xx.array()).sin()*(2.0*M_PI*yy.array()).sin();
  y.array() += -a*(2.0*M_PI*xx.array()).sin()*(M_PI*yy.array()).sin();

#endif

  // vol geofacs
  MatrixXd Drq = ref_data->Vq * ref_data->Dr;
  MatrixXd Dsq = ref_data->Vq * ref_data->Ds;
  MatrixXd xr, yr, xs, ys;
  xr = Drq * x;
  yr = Drq * y;
  xs = Dsq * x;
  ys = Dsq * y;

  MatrixXd rxJ = ys.array();
  MatrixXd sxJ = -yr.array();
  MatrixXd ryJ = -xs.array();
  MatrixXd syJ = xr.array();
  MatrixXd J = xr.array() * ys.array() - yr.array() * xs.array();

  // surface geofacs
  MatrixXd Drfq = ref_data->Vfq * ref_data->Dr;
  MatrixXd Dsfq = ref_data->Vfq * ref_data->Ds;

  xr = Drfq * x;
  yr = Drfq * y;
  xs = Dsfq * x;
  ys = Dsfq * y;

  MatrixXd rxJf = ys.array();
  MatrixXd sxJf = -yr.array();
  MatrixXd ryJf = -xs.array();
  MatrixXd syJf = xr.array();

  MatrixXd Jf = xr.array() * ys.array() - yr.array() * xs.array();

  MatrixXd nrJ = ref_data->nrJ.replicate(1, EToVX.cols());
  MatrixXd nsJ = ref_data->nsJ.replicate(1, EToVX.cols());

  MatrixXd nxJ = rxJf.array() * nrJ.array() + sxJf.array() * nsJ.array();
  MatrixXd nyJ = ryJf.array() * nrJ.array() + syJf.array() * nsJ.array();

  MatrixXd nx = nxJ.array() / Jf.array();
  MatrixXd ny = nyJ.array() / Jf.array();

  MatrixXd sJ = (nx.array().pow(2) + ny.array().pow(2)).array().sqrt();
  nx = nx.array() / sJ.array();
  ny = ny.array() / sJ.array();
  sJ = sJ.array() * Jf.array();

  geo_elem_data *geo = new geo_elem_data;
  geo->xq = ref_data->Vq * x;
  geo->yq = ref_data->Vq * y;

  geo->xf = ref_data->Vfq * x;
  geo->yf = ref_data->Vfq * y;

  geo->rxJ = rxJ;
  geo->sxJ = sxJ;
  geo->ryJ = ryJ;
  geo->syJ = syJ;

  geo->rxJf = rxJf;
  geo->sxJf = sxJf;
  geo->ryJf = ryJf;
  geo->syJf = syJf;

  geo->nxJ = nxJ;
  geo->nyJ = nyJ;

  MatrixXd VqPq = (ref_data->Vq)*(ref_data->Pq);
  geo->J = VqPq*J; // to ensure conservation
  //geo->J = J;

  geo->sJ = sJ;
  return geo;
}

geo_elem_data *build_geofacs_3D(ref_elem_data *ref_data,
                                const Ref<MatrixXd> EToVX)
{
  VectorXd r = ref_data->r;
  VectorXd s = ref_data->s;
  VectorXd t = ref_data->t;
  int N = ref_data->N;

#if 0
  double a = .05;
  VectorXd dr = Eigen::pow(r.array() + s.array(), N);
  VectorXd ds = Eigen::pow(s.array() + t.array(), N);
  VectorXd dt = Eigen::pow(r.array() + t.array(), N);
  VectorXd x = r - a / 1.0 * dr;
  VectorXd y = s + a / 2.0 * ds;
  VectorXd z = t + a / 3.0 * dt;
#else

  // TODO double check the order here
  MatrixXd vxa = EToVX.row(0);
  MatrixXd vya = EToVX.row(1);
  MatrixXd vza = EToVX.row(2);

  MatrixXd vxb = EToVX.row(3);
  MatrixXd vyb = EToVX.row(4);
  MatrixXd vzb = EToVX.row(5);

  MatrixXd vxc = EToVX.row(6);
  MatrixXd vyc = EToVX.row(7);
  MatrixXd vzc = EToVX.row(8);

  MatrixXd vxd = EToVX.row(9);
  MatrixXd vyd = EToVX.row(10);
  MatrixXd vzd = EToVX.row(11);

  VectorXd one = VectorXd::Ones(r.size());
  MatrixXd x = 0.5 * (-(one + r + s + t) * vxa + (one + r) * vxb +
                      (one + s) * vxc + (one + t) * vxd);
  MatrixXd y = 0.5 * (-(one + r + s + t) * vya + (one + r) * vyb +
                      (one + s) * vyc + (one + t) * vyd);
  MatrixXd z = 0.5 * (-(one + r + s + t) * vza + (one + r) * vzb +
                      (one + s) * vzc + (one + t) * vzd);

#endif


#if 0
  printf("Adding curvilinear perturbation for isentropic vortex\n");
  // add curvilinear perturbation to [0,10] x [0,20] x [0,10]
  MatrixXd xx = x.array()/10.0;
  MatrixXd yy = y.array()/20.0;
  MatrixXd zz = z.array()/10.0;
  double a = 0.5;
  x.array() +=      a*(M_PI*xx.array()).sin()*(2.0*M_PI*yy.array()).sin()*(M_PI*zz.array()).sin();
  y.array() += -2.0*a*(2.0*M_PI*xx.array()).sin()*(M_PI*yy.array()).sin()*(2.0*M_PI*zz.array()).sin();;
  z.array() +=      a*(M_PI*xx.array()).sin()*(2.0*M_PI*yy.array()).sin()*(M_PI*zz.array()).sin();

#endif

#if 0
  printf("Adding curvilinear perturbation for TG vortex\n");
  // add curvilinear perturbation to [-pi,pi]^3
  MatrixXd xx = x.array();
  MatrixXd yy = y.array();
  MatrixXd zz = z.array();
  double a = 0.5;
  a = .125; // for really coarse meshes...
  x.array() += a*x.array().sin()*y.array().sin()*z.array().sin();
  y.array() += a*x.array().sin()*y.array().sin()*z.array().sin();
  z.array() += a*x.array().sin()*y.array().sin()*z.array().sin();

#endif

  // vol geofacs
  MatrixXd Drq = ref_data->Vq * ref_data->Dr;
  MatrixXd Dsq = ref_data->Vq * ref_data->Ds;
  MatrixXd Dtq = ref_data->Vq * ref_data->Dt;
  MatrixXd xr, yr, zr, xs, ys, zs, xt, yt, zt;
  xr = Drq * x;
  yr = Drq * y;
  zr = Drq * z;
  xs = Dsq * x;
  ys = Dsq * y;
  zs = Dsq * z;
  xt = Dtq * x;
  yt = Dtq * y;
  zt = Dtq * z;

  MatrixXd rxJ = ys.array() * zt.array() - zs.array() * yt.array();
  MatrixXd sxJ = yt.array() * zr.array() - zt.array() * yr.array();
  MatrixXd txJ = yr.array() * zs.array() - zr.array() * ys.array();

  MatrixXd ryJ = xt.array() * zs.array() - xs.array() * zt.array();
  MatrixXd syJ = xr.array() * zt.array() - xt.array() * zr.array();
  MatrixXd tyJ = xs.array() * zr.array() - zs.array() * xr.array();

  MatrixXd rzJ = xs.array() * yt.array() - xt.array() * ys.array();
  MatrixXd szJ = xt.array() * yr.array() - xr.array() * yt.array();
  MatrixXd tzJ = xr.array() * ys.array() - xs.array() * yr.array();

  MatrixXd J =
      xr.array() * (ys.array() * zt.array() - zs.array() * yt.array()) -
      yr.array() * (xs.array() * zt.array() - zs.array() * xt.array()) +
      zr.array() * (xs.array() * yt.array() - ys.array() * xt.array());

  // surface geofacs
  MatrixXd Drfq = ref_data->Vfq * ref_data->Dr;
  MatrixXd Dsfq = ref_data->Vfq * ref_data->Ds;
  MatrixXd Dtfq = ref_data->Vfq * ref_data->Dt;

  xr = Drfq * x;
  yr = Drfq * y;
  zr = Drfq * z;
  xs = Dsfq * x;
  ys = Dsfq * y;
  zs = Dsfq * z;
  xt = Dtfq * x;
  yt = Dtfq * y;
  zt = Dtfq * z;

  MatrixXd rxJf = ys.array() * zt.array() - zs.array() * yt.array();
  MatrixXd sxJf = yt.array() * zr.array() - zt.array() * yr.array();
  MatrixXd txJf = yr.array() * zs.array() - zr.array() * ys.array();

  MatrixXd ryJf = xt.array() * zs.array() - xs.array() * zt.array();
  MatrixXd syJf = xr.array() * zt.array() - xt.array() * zr.array();
  MatrixXd tyJf = xs.array() * zr.array() - zs.array() * xr.array();

  MatrixXd rzJf = xs.array() * yt.array() - xt.array() * ys.array();
  MatrixXd szJf = xt.array() * yr.array() - xr.array() * yt.array();
  MatrixXd tzJf = xr.array() * ys.array() - xs.array() * yr.array();

  MatrixXd Jf =
      xr.array() * (ys.array() * zt.array() - zs.array() * yt.array()) -
      yr.array() * (xs.array() * zt.array() - zs.array() * xt.array()) +
      zr.array() * (xs.array() * yt.array() - ys.array() * xt.array());

  MatrixXd nrJ = ref_data->nrJ.replicate(1, EToVX.cols());
  MatrixXd nsJ = ref_data->nsJ.replicate(1, EToVX.cols());
  MatrixXd ntJ = ref_data->ntJ.replicate(1, EToVX.cols());

  MatrixXd nxJ = rxJf.array() * nrJ.array() + sxJf.array() * nsJ.array() +
                 txJf.array() * ntJ.array();
  MatrixXd nyJ = ryJf.array() * nrJ.array() + syJf.array() * nsJ.array() +
                 tyJf.array() * ntJ.array();
  MatrixXd nzJ = rzJf.array() * nrJ.array() + szJf.array() * nsJ.array() +
                 tzJf.array() * ntJ.array();

  MatrixXd nx = nxJ.array() / Jf.array();
  MatrixXd ny = nyJ.array() / Jf.array();
  MatrixXd nz = nzJ.array() / Jf.array();
  MatrixXd sJ = (nx.array().pow(2) + ny.array().pow(2) + nz.array().pow(2))
                    .array()
                    .sqrt();
  nx = nx.array() / sJ.array();
  ny = ny.array() / sJ.array();
  nz = nz.array() / sJ.array();
  sJ = sJ.array() * Jf.array();

#if 1 // interpolated conservative curl form from Kopriva

  int Ngeo = N+1;
  ref_elem_data *ref_data_N2 = build_ref_ops_3D(
      Ngeo, 2 * N + 2, 2 * N + 2); // build VDM and Dmats for deg N+1
  MatrixXd Dr = ref_data_N2->Dr;
  MatrixXd Ds = ref_data_N2->Ds;
  MatrixXd Dt = ref_data_N2->Dt;

  // interpolation matrices
  VectorXd r2, s2, t2;
  Nodes3D(Ngeo, r2, s2, t2);
  MatrixXd V1 = Vandermonde3D(N, r, s, t);
  MatrixXd V12tmp = Vandermonde3D(N, r2, s2, t2);
  MatrixXd V12 = mrdivide(V12tmp, V1);

  MatrixXd V2 = Vandermonde3D(Ngeo, r2, s2, t2);
  MatrixXd V21tmp = Vandermonde3D(Ngeo, r, s, t);
  MatrixXd V21 = mrdivide(V21tmp, V2);

  MatrixXd x2 = V12 * x;
  MatrixXd y2 = V12 * y;
  MatrixXd z2 = V12 * z;
  xr = Dr * x2;
  yr = Dr * y2;
  zr = Dr * z2;
  xs = Ds * x2;
  ys = Ds * y2;
  zs = Ds * z2;
  xt = Dt * x2;
  yt = Dt * y2;
  zt = Dt * z2;

  // MatrixXd rxJ, sxJ, txJ, ryJ, syJ, tyJ, rzJ, szJ, tzJ;
  rxJ = Dt * (ys.array() * z2.array()).matrix() -
        Ds * (yt.array() * z2.array()).matrix();
  sxJ = Dr * (yt.array() * z2.array()).matrix() -
        Dt * (yr.array() * z2.array()).matrix();
  txJ = Ds * (yr.array() * z2.array()).matrix() -
        Dr * (ys.array() * z2.array()).matrix();

  ryJ = -(Dt * (xs.array() * z2.array()).matrix() -
          Ds * (xt.array() * z2.array()).matrix());
  syJ = -(Dr * (xt.array() * z2.array()).matrix() -
          Dt * (xr.array() * z2.array()).matrix());
  tyJ = -(Ds * (xr.array() * z2.array()).matrix() -
          Dr * (xs.array() * z2.array()).matrix());

  rzJ = -(Dt * (ys.array() * x2.array()).matrix() -
          Ds * (yt.array() * x2.array()).matrix());
  szJ = -(Dr * (yt.array() * x2.array()).matrix() -
          Dt * (yr.array() * x2.array()).matrix());
  tzJ = -(Ds * (yr.array() * x2.array()).matrix() -
          Dr * (ys.array() * x2.array()).matrix());

  // interp back to degree N
  rxJ = V21 * rxJ;
  sxJ = V21 * sxJ;
  txJ = V21 * txJ;
  ryJ = V21 * ryJ;
  syJ = V21 * syJ;
  tyJ = V21 * tyJ;
  rzJ = V21 * rzJ;
  szJ = V21 * szJ;
  tzJ = V21 * tzJ;

  // check satisfaction of GCL
  MatrixXd divx = ((ref_data->Dr*rxJ + ref_data->Ds*sxJ + ref_data->Dt*txJ).array().abs()).matrix();
  MatrixXd divy = ((ref_data->Dr*ryJ + ref_data->Ds*syJ + ref_data->Dt*tyJ).array().abs()).matrix();
  MatrixXd divz = ((ref_data->Dr*rzJ + ref_data->Ds*szJ + ref_data->Dt*tzJ).array().abs()).matrix();
  printf("divxyz = %g, %g, %g\n", divx.sum(), divy.sum(), divz.sum());

  // compute normals
  MatrixXd Vfq = ref_data->Vfq;
  rxJf = Vfq * rxJ;
  sxJf = Vfq * sxJ;
  txJf = Vfq * txJ;
  ryJf = Vfq * ryJ;
  syJf = Vfq * syJ;
  tyJf = Vfq * tyJ;
  rzJf = Vfq * rzJ;
  szJf = Vfq * szJ;
  tzJf = Vfq * tzJ;
  nxJ = rxJf.array() * nrJ.array() + sxJf.array() * nsJ.array() +
        txJf.array() * ntJ.array();
  nyJ = ryJf.array() * nrJ.array() + syJf.array() * nsJ.array() +
        tyJf.array() * ntJ.array();
  nzJ = rzJf.array() * nrJ.array() + szJf.array() * nsJ.array() +
        tzJf.array() * ntJ.array();

  // eval at qpts
  MatrixXd Vq = ref_data->Vq;
  rxJ = Vq * rxJ;
  sxJ = Vq * sxJ;
  txJ = Vq * txJ;
  ryJ = Vq * ryJ;
  syJ = Vq * syJ;
  tyJ = Vq * tyJ;
  rzJ = Vq * rzJ;
  szJ = Vq * szJ;
  tzJ = Vq * tzJ;

#endif

  geo_elem_data *geo = new geo_elem_data;
  geo->xq = ref_data->Vq * x;
  geo->yq = ref_data->Vq * y;
  geo->zq = ref_data->Vq * z;

  geo->xf = ref_data->Vfq * x;
  geo->yf = ref_data->Vfq * y;
  geo->zf = ref_data->Vfq * z;

  geo->rxJ = rxJ;
  geo->sxJ = sxJ;
  geo->txJ = txJ;
  geo->ryJ = ryJ;
  geo->syJ = syJ;
  geo->tyJ = tyJ;
  geo->rzJ = rzJ;
  geo->szJ = szJ;
  geo->tzJ = tzJ;

  geo->rxJf = rxJf;
  geo->sxJf = sxJf;
  geo->txJf = txJf;
  geo->ryJf = ryJf;
  geo->syJf = syJf;
  geo->tyJf = tyJf;
  geo->rzJf = rzJf;
  geo->szJf = szJf;
  geo->tzJf = tzJf;

  //  cout << "nxJ = "<<endl << nxJ << endl;
  //  cout << "nyJ = "<<endl << nyJ << endl;
  //  cout << "nzJ = "<<endl << nzJ << endl;

  geo->nxJ = nxJ;
  geo->nyJ = nyJ;
  geo->nzJ = nzJ;

  MatrixXd VqPq = (ref_data->Vq)*(ref_data->Pq);
  geo->J = VqPq*J; // to ensure conservation, project onto deg N polynomials
  //  geo->J = J;
  geo->sJ = sJ;
  return geo;
}

map_elem_data *build_maps_2D(ref_elem_data *ref_data,
                             const Ref<MatrixXu32> EToE,
                             const Ref<MatrixXu8> EToF,
                             const Ref<MatrixXu8> EToO)
{
  const int N = ref_data->N;
  const int Np = (int)ref_data->r.size();
  const int Nfp = N + 1;
  const int Nfq = (int)ref_data->ref_rfq.size();
  const int Nfaces = ref_data->Nfaces;
  const int No = 2; // number of orientations
  const double NODETOL = Eigen::NumTraits<double>::epsilon() * 10;

  MatrixXd rM = ref_data->ref_rfq;
  MatrixXd rP = ref_data->ref_rfq;
  VectorXd ones = MatrixXd::Ones(Nfq, 1);

  MatrixXu32 OmapP(Nfq, No);

  for (int o = 0; o < No; ++o)
  {
    switch (o)
    {
    case 0:
      rP = rM;
      break;
    case 1:
      rP = -rM;
      break;
    default:
      cerr << "Missing node" << endl;
      abort();
    }

    for (int i = 0; i < Nfq; ++i)
    {
      int j;
      for (j = 0; j < Nfq; ++j)
      {
        if (fabs(rM(i) - rP(j)) < NODETOL)
        {
          OmapP(i, o) = j;
          break;
        }
      }
      if (j == Nfq)
      {
        cerr << "Dropped a node: (" << rM(i) << ", " << rP(i) << ")" << endl;
        abort();
      }
    }
  }

  const uint32_t E = (uint32_t)EToE.cols();
  MatrixXu32 mapPq(Nfq * Nfaces, E);

  for (uint32_t e1 = 0; e1 < E; ++e1)
  {
    for (int f1 = 0; f1 < Nfaces; ++f1)
    {
      const uint32_t e2 = EToE(f1, e1);
      const uint8_t f2 = EToF(f1, e1);
      const uint8_t o2 = EToO(f1, e1);

      const uint32_t shift = e2 * Nfq * Nfaces + f2 * Nfq;
      for (int n = 0; n < Nfq; ++n)
        mapPq(f1 * Nfq + n, e1) = shift + OmapP(n, o2);
    }
  }

  MatrixXu32 fmask(Nfp, Nfaces);
  VectorXd r = ref_data->r;
  VectorXd s = ref_data->s;

  {
    int f = 0;
    int j = 0;
    for (int i = 0; i < Np; ++i)
      if (fabs(1.0 + s(i)) < NODETOL)
        fmask(j++, f) = i;
    if (j != Nfp)
    {
      cerr << "Wrong number of face nodes found" << endl;
      abort();
    }

    f = 1;
    j = 0;
    for (int i = 0; i < Np; ++i)
      if (fabs(r(i) + s(i)) < NODETOL)
        fmask(j++, f) = i;
    if (j != Nfp)
    {
      cerr << "Wrong number of face nodes found" << endl;
      abort();
    }

    f = 2;
    j = 0;
    for (int i = 0; i < Np; ++i)
      if (fabs(1.0 + r(i)) < NODETOL)
        fmask(j++, f) = i;
    if (j != Nfp)
    {
      cerr << "Wrong number of face nodes found" << endl;
      abort();
    }
  }

  map_elem_data *map = new map_elem_data;
  map->mapPq = mapPq;
  map->fmask = fmask;
  return map;
}

map_elem_data *build_maps_3D(ref_elem_data *ref_data,
                             const Ref<MatrixXu32> EToE,
                             const Ref<MatrixXu8> EToF,
                             const Ref<MatrixXu8> EToO)
{
  const int N = ref_data->N;
  const int Np = (int)ref_data->r.size();
  const int Nfp = (N + 1) * (N + 2) / 2;
  const int Nfq = (int)ref_data->ref_rfq.size();
  const int Nfaces = ref_data->Nfaces;
  const int No = 6; // number of orientations
  const double NODETOL = Eigen::NumTraits<double>::epsilon() * 10;

  MatrixXd rM = ref_data->ref_rfq;
  MatrixXd sM = ref_data->ref_sfq;
  MatrixXd rP = ref_data->ref_rfq;
  MatrixXd sP = ref_data->ref_sfq;
  VectorXd ones = MatrixXd::Ones(Nfq, 1);

  MatrixXu32 OmapP(Nfq, No);

  for (int o = 0; o < No; ++o)
  {
    switch (o)
    {
    case 0:
      rP = rM;
      sP = sM;
      break;
    case 1:
      rP = sM;
      sP = -rM - sM - ones;
      break;
    case 2:
      rP = -rM - sM - ones;
      sP = rM;
      break;
    case 3:
      rP = -rM - sM - ones;
      sP = sM;
      break;
    case 4:
      rP = rM;
      sP = -rM - sM - ones;
      break;
    case 5:
      rP = sM;
      sP = rM;
      break;
    default:
      cerr << "Missing node" << endl;
      abort();
    }

    for (int i = 0; i < Nfq; ++i)
    {
      int j;
      for (j = 0; j < Nfq; ++j)
      {
        if (hypot(rM(i) - rP(j), sM(i) - sP(j)) < NODETOL)
        {
          OmapP(i, o) = j;
          break;
        }
      }
      if (j == Nfq)
      {
        cerr << "Dropped a node: (" << rM(i) << ", " << rP(i) << ")" << endl;
        abort();
      }
    }
  }

  const uint32_t E = (uint32_t)EToE.cols();
  MatrixXu32 mapPq(Nfq * Nfaces, E);

  for (uint32_t e1 = 0; e1 < E; ++e1)
  {
    for (int f1 = 0; f1 < Nfaces; ++f1)
    {
      const uint32_t e2 = EToE(f1, e1);
      const uint8_t f2 = EToF(f1, e1);
      const uint8_t o2 = EToO(f1, e1);

      const uint32_t shift = e2 * Nfq * Nfaces + f2 * Nfq;
      for (int n = 0; n < Nfq; ++n)
        mapPq(f1 * Nfq + n, e1) = shift + OmapP(n, o2);
    }
  }

  MatrixXu32 fmask(Nfp, Nfaces);
  VectorXd r = ref_data->r;
  VectorXd s = ref_data->s;
  VectorXd t = ref_data->t;

  {
    int f = 0;
    int j = 0;
    for (int i = 0; i < Np; ++i)
      if (fabs(1.0 + t(i)) < NODETOL)
        fmask(j++, f) = i;
    if (j != Nfp)
    {
      cerr << "Wrong number of face nodes found" << endl;
      abort();
    }

    f = 1;
    j = 0;
    for (int i = 0; i < Np; ++i)
      if (fabs(1.0 + s(i)) < NODETOL)
        fmask(j++, f) = i;
    if (j != Nfp)
    {
      cerr << "Wrong number of face nodes found" << endl;
      abort();
    }

    f = 2;
    j = 0;
    for (int i = 0; i < Np; ++i)
      if (fabs(1.0 + r(i) + s(i) + t(i)) < NODETOL)
        fmask(j++, f) = i;
    if (j != Nfp)
    {
      cerr << "Wrong number of face nodes found" << endl;
      abort();
    }

    f = 3;
    j = 0;
    for (int i = 0; i < Np; ++i)
      if (fabs(1.0 + r(i)) < NODETOL)
        fmask(j++, f) = i;
    if (j != Nfp)
    {
      cerr << "Wrong number of face nodes found" << endl;
      abort();
    }
  }

  map_elem_data *map = new map_elem_data;
  map->mapPq = mapPq;
  map->fmask = fmask;
  return map;
}

// =================== begin matlab codes =======================

// d = degree of basis
VectorXd JacobiP(VectorXd x, double alpha, double beta, int d)
{
  int Nx = (int)x.rows();
  // cout << "Nx = " << Nx << ", x = " << x << endl;
  VectorXd P(Nx, 1);
  MatrixXd PL(Nx, d + 1);

  // initial P_0 and P_1
  double gamma0 = pow(2, alpha + beta + 1) / (alpha + beta + 1) *
                  tgamma(alpha + 1) * tgamma(beta + 1) /
                  tgamma(alpha + beta + 1);
  for (int i = 0; i < Nx; i++)
    PL(i, 0) = 1.0 / sqrt(gamma0);
  if (d == 0)
    return PL;

  double gamma1 = (alpha + 1.) * (beta + 1.) / (alpha + beta + 3.) * gamma0;
  for (int i = 0; i < Nx; i++)
    PL(i, 1) =
        ((alpha + beta + 2) * x(i) / 2.0 + (alpha - beta) / 2) / sqrt(gamma1);
  if (d == 1)
  {
    for (int i = 0; i < Nx; i++)
      P(i) = PL(i, 1);
    return P;
  }

  // repeat value in recurrence
  double aold = 2.0 / (2 + alpha + beta) *
                sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3));

  // recurrence
  for (int i = 1; i <= d - 1; i++)
  {
    double h1 = 2 * i + alpha + beta;
    // clang-format off
    double anew =
        2 / (h1 + 2) * sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) *
                            (i + 1 + beta) / (h1 + 1) / (h1 + 3));
    // clang-format on
    double bnew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2);
    // cout << "anew = " << anew << ", bnew = " << bnew << endl;
    for (int j = 1; j <= Nx; j++)
    {
      PL(j - 1, i - 1 + 2) =
          1.0 / anew *
          (-aold * PL(j - 1, i - 1) + (x(j - 1) - bnew) * PL(j - 1, i - 1 + 1));
    }
    aold = anew;
  }

  for (int i = 0; i < Nx; i++)
  {
    P(i) = PL(i, d);
  }
  return P;
}

MatrixXd Vandermonde1D(int N, VectorXd r)
{
  int Np = (N + 1);
  MatrixXd Vout(r.rows(), Np);

  for (int i = 0; i <= N; ++i)
  {
    Vout.col(i) = JacobiP(r, 0, 0, i);
  }
  return Vout;
}

VectorXd GradJacobiP(VectorXd x, double alpha, double beta, int p)
{

  /// function [dP] = gradJacobiP(z, alpha, beta, N);
  /// Purpose: Evaluate the derivative of the orthonormal Jacobi
  ///   polynomial of type (alpha,beta)>-1, at points x
  ///          for order N and returns dP[1:length(xp))]
  int Nx = (int)x.rows();
  VectorXd dP(Nx);

  if (p == 0)
    dP = 0.0 * dP;
  else
  {
    dP = sqrt(p * (p + alpha + beta + 1)) *
         JacobiP(x, alpha + 1, beta + 1, p - 1);
  }
  return dP;
}

// ================= 2D routines ===================

void rstoab(VectorXd r, VectorXd s, VectorXd &a, VectorXd &b)
{

  int Npts = (int)r.rows();
  a.resize(Npts);
  b.resize(Npts);
  double tol = 1e-8;

  for (int n = 0; n < Npts; ++n)
  {
    if (fabs(s(n) - 1) > tol)
      a(n) = 2 * (1 + r(n)) / (1 - s(n)) - 1;
    else
      a(n) = -1;
    b(n) = s(n);
  }
}

VectorXd Simplex2DP(VectorXd a, VectorXd b, int i, int j)
{

  VectorXd h1 = JacobiP(a, 0, 0, i);
  VectorXd h2 = JacobiP(b, 2 * i + 1, 0, j);

  VectorXd bpow = Eigen::pow(1 - b.array(), i);

  VectorXd P = sqrt(2) * h1.array() * h2.array() * bpow.array();
  return P;
}

MatrixXd Vandermonde2D(int N, VectorXd r, VectorXd s)
{
  int Np = (N + 1) * (N + 2) / 2;
  MatrixXd Vout(r.rows(), Np);

  VectorXd a, b, c;
  rstoab(r, s, a, b);
  int sk = 0;
  for (int i = 0; i <= N; ++i)
  {
    for (int j = 0; j <= N - i; ++j)
    {
      Vout.col(sk) = Simplex2DP(a, b, i, j);
      sk = sk + 1;
    }
  }
  return Vout;
}

void GradSimplex2DP(VectorXd a, VectorXd b, int id, int jd, VectorXd &V2Dr,
                    VectorXd &V2Ds)
{

  VectorXd fa, dfa, gb, dgb;
  fa = JacobiP(a, 0, 0, id);
  dfa = GradJacobiP(a, 0, 0, id);
  gb = JacobiP(b, 2 * id + 1, 0, jd);
  dgb = GradJacobiP(b, 2 * id + 1, 0, jd);

  // r-derivative
  int Np = (int)a.rows();
  V2Dr.resize(Np);
  V2Dr.array() = dfa.array() * gb.array();
  if (id > 0)
    V2Dr.array() *= Eigen::pow(0.5 * (1 - b.array()), id - 1);

  // s-derivative
  V2Ds.resize(Np);
  V2Ds.array() = dfa.array() * gb.array() * (.5 * (1 + a.array()));
  if (id > 0)
    V2Ds.array() *= Eigen::pow(.5 * (1 - b.array()), id - 1);

  VectorXd tmp = dgb.array() * Eigen::pow(.5 * (1 - b.array()), id);
  if (id > 0)
    tmp.array() -=
        .5 * id * gb.array() * Eigen::pow(.5 * (1 - b.array()), id - 1);

  V2Ds.array() += fa.array() * tmp.array();

  // normalize
  V2Dr = V2Dr * pow(2, id + .5);
  V2Ds = V2Ds * pow(2, id + .5);
}
void GradVandermonde2D(int N, VectorXd r, VectorXd s, MatrixXd &V2Dr,
                       MatrixXd &V2Ds)
{

  int Npts = (int)r.rows();
  int Np = (N + 1) * (N + 2) / 2;
  V2Dr.resize(Npts, Np);
  V2Ds.resize(Npts, Np);

  // find tensor-product coordinates
  VectorXd a, b;
  rstoab(r, s, a, b);

  // Initialize matrices
  int sk = 0;
  for (int i = 0; i <= N; ++i)
  {
    for (int j = 0; j <= N - i; ++j)
    {
      VectorXd Vr, Vs;
      GradSimplex2DP(a, b, i, j, Vr, Vs);
      V2Dr.col(sk) = Vr;
      V2Ds.col(sk) = Vs;
      sk = sk + 1;
    }
  }
}

// ========================== 3D ===========================

void rsttoabc(VectorXd r, VectorXd s, VectorXd t, VectorXd &a, VectorXd &b,
              VectorXd &c)
{

  int Np = (int)r.rows();
  a.resize(Np);
  b.resize(Np);
  c.resize(Np);
  double tol = 1e-8;
  for (int n = 0; n < Np; ++n)
  {
    if (fabs(s(n) + t(n)) > tol)
      a(n) = 2 * (1 + r(n)) / (-s(n) - t(n)) - 1;
    else
      a(n) = -1;

    if (fabs(t(n) - 1) > tol)
      b(n) = 2 * (1 + s(n)) / (1 - t(n)) - 1;
    else
      b(n) = -1;

    c(n) = t(n);
  }
}

VectorXd Simplex3DP(VectorXd a, VectorXd b, VectorXd c, int i, int j, int k)
{

  // function [P] = Simplex3DP(a,b,c,i,j,k);
  // Purpose : Evaluate 3D orthonormal polynomial
  //           on simplex at (a,b,c) of order (i,j,k).

  VectorXd h1 = JacobiP(a, 0, 0, i);
  VectorXd h2 = JacobiP(b, 2 * i + 1, 0, j);
  VectorXd h3 = JacobiP(c, 2 * (i + j) + 2, 0, k);

  VectorXd bpow = Eigen::pow(1 - b.array(), i);
  VectorXd cpow = Eigen::pow(1 - c.array(), i + j);

  VectorXd P = 2 * sqrt(2) * h1.array() * h2.array() * bpow.array() *
               h3.array() * cpow.array();
  return P;
}

void GradSimplex3DP(VectorXd a, VectorXd b, VectorXd c, int id, int jd, int kd,
                    VectorXd &V3Dr, VectorXd &V3Ds, VectorXd &V3Dt)
{

  // function [V3Dr, V3Ds, V3Dt] = GradSimplex3DP(a,b,c,id,jd,kd)
  // Purpose: Return the derivatives of the modal basis (id,jd,kd)
  //          on the 3D simplex at (a,b,c)

  VectorXd fa, dfa, gb, dgb, hc, dhc;
  fa = JacobiP(a, 0, 0, id);
  dfa = GradJacobiP(a, 0, 0, id);
  gb = JacobiP(b, 2 * id + 1, 0, jd);
  dgb = GradJacobiP(b, 2 * id + 1, 0, jd);
  hc = JacobiP(c, 2 * (id + jd) + 2, 0, kd);
  dhc = GradJacobiP(c, 2 * (id + jd) + 2, 0, kd);

  // r-derivative
  int Np = (int)a.rows();
  V3Dr.resize(Np);
  V3Dr.array() = dfa.array() * (gb.array() * hc.array());
  if (id > 0)
    V3Dr.array() *= Eigen::pow(0.5 * (1 - b.array()), id - 1);

  if (id + jd > 0)
    V3Dr.array() *= Eigen::pow(0.5 * (1 - c.array()), id + jd - 1);

  // s-derivative
  V3Ds.resize(Np);
  V3Ds.array() = 0.5 * (1 + a.array()) * V3Dr.array();
  VectorXd tmp = dgb.array() * Eigen::pow(0.5 * (1 - b.array()), id);
  if (id > 0)
    tmp.array() +=
        (-0.5 * id) * (gb.array() * Eigen::pow(0.5 * (1 - b.array()), id - 1));
  if (id + jd > 0)
    tmp.array() *= Eigen::pow(0.5 * (1 - c.array()), id + jd - 1);
  tmp.array() = fa.array() * (tmp.array() * hc.array());
  V3Ds.array() += tmp.array();

  // t-derivative
  V3Dt.resize(Np);
  V3Dt.array() = 0.5 * (1 + a.array()) * V3Dr.array() +
                 0.5 * (1 + b.array()) * tmp.array();
  tmp.array() = dhc.array() * Eigen::pow(0.5 * (1 - c.array()), id + jd);
  if (id + jd > 0)
    tmp.array() +=
        -0.5 * (id + jd) *
        (hc.array() * Eigen::pow(0.5 * (1 - c.array()), id + jd - 1));
  tmp.array() = fa.array() * (gb.array() * tmp.array());
  tmp.array() *= Eigen::pow(0.5 * (1 - b.array()), id);
  V3Dt.array() = V3Dt.array() + tmp.array();

  // normalize
  V3Dr = V3Dr * pow(2, 2 * id + jd + 1.5);
  V3Ds = V3Ds * pow(2, 2 * id + jd + 1.5);
  V3Dt = V3Dt * pow(2, 2 * id + jd + 1.5);
}

MatrixXd Vandermonde3D(int N, VectorXd r, VectorXd s, VectorXd t)
{
  /// function [Vout] = Voutandermonde(N,r,s,t,Vout)
  /// Purpose : Initialize the gradient of the modal basis (i,j,k)
  /// at (r,s,t) at order p

  int Np = (N + 1) * (N + 2) * (N + 3) / 6;
  MatrixXd Vout(r.rows(), Np);

  /// find tensor-product coordinates
  VectorXd a, b, c;
  rsttoabc(r, s, t, a, b, c);

  /// initialize matrices
  int sk = 0;
  for (int i = 0; i <= N; ++i)
  {
    for (int j = 0; j <= N - i; ++j)
    {
      for (int k = 0; k <= N - i - j; ++k)
      {
        Vout.col(sk) = Simplex3DP(a, b, c, i, j, k);
        sk++;
      }
    }
  }
  return Vout;
}

void GradVandermonde3D(int N, VectorXd r, VectorXd s, VectorXd t,
                       MatrixXd &V3Dr, MatrixXd &V3Ds, MatrixXd &V3Dt)
{

  int Npts = (int)r.rows();
  int Np = (N + 1) * (N + 2) * (N + 3) / 6;
  V3Dr.resize(Npts, Np);
  V3Ds.resize(Npts, Np);
  V3Dt.resize(Npts, Np);

  // find tensor-product coordinates
  VectorXd a, b, c;
  rsttoabc(r, s, t, a, b, c);

  // Initialize matrices
  int sk = 0;
  for (int i = 0; i <= N; ++i)
  {
    for (int j = 0; j <= N - i; ++j)
    {
      for (int k = 0; k <= N - i - j; ++k)
      {
        VectorXd Vr, Vs, Vt;
        GradSimplex3DP(a, b, c, i, j, k, Vr, Vs, Vt);
        V3Dr.col(sk) = Vr;
        V3Ds.col(sk) = Vs;
        V3Dt.col(sk) = Vt;
        sk = sk + 1;
      }
    }
  }
}

// =============== Hierarchical basis ==================

void barytors(VectorXd L1, VectorXd L2, VectorXd L3, VectorXd &r, VectorXd &s)
{
  VectorXd v1(2);
  v1 << -1, -1;
  VectorXd v2(2);
  v2 << 1, -1;
  VectorXd v3(2);
  v3 << -1, 1;

  int Npts = (int)L1.rows();
  r.resize(Npts);
  s.resize(Npts);
  for (int i = 0; i < Npts; ++i)
  {
    VectorXd XY = v1 * L1(i) + v2 * L2(i) + v3 * L3(i);
    r(i) = XY(1);
    s(i) = XY(2);
  }
}

// =============== Bernstein basis ================

unsigned int factorial_ratio(int n1, int n2)
{
  unsigned int val = 1;
  for (int i = n1; i > n2; --i)
  {
    val *= i;
  }
  return val;
}

unsigned int factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

unsigned int nchoosek(int n, int k)
{
  return factorial(n) / (factorial(n - k) * factorial(k));
}

MatrixXd Bern1D(int N, VectorXd r)
{
  VectorXd x = .5 * (1.0 + r.array());
  int Np = (N + 1);
  MatrixXd V(r.rows(), Np);
  V.fill(0.0);
  for (int i = 0; i < Np; ++i)
  {
    V.col(i) = nchoosek(N, i) * x.array().pow(i) *
               (1.0 - x.array()).array().pow(N - i);
  }
  return V;
}

MatrixXd BernTri(int N, VectorXd r, VectorXd s)
{
  VectorXd L1 = -(r.array() + s.array()) / 2;
  VectorXd L2 = (1 + r.array()) / 2;
  VectorXd L3 = (1 + s.array()) / 2;

  int Np = (N + 1) * (N + 2) / 2;
  MatrixXd V(r.rows(), Np);
  int sk = 0;
  for (int k = 0; k <= N; ++k)
  {
    for (int j = 0; j <= N - k; ++j)
    {
      int i = N - j - k;
      double C =
          (double)factorial(N) / (factorial(i) * factorial(j) * factorial(k));
      VectorXd L1i = Eigen::pow(L1.array(), i);
      VectorXd L2j = Eigen::pow(L2.array(), j);
      VectorXd L3k = Eigen::pow(L3.array(), k);
      V.col(sk) = C * L1i.array() * L2j.array() * L3k.array();
      sk = sk + 1;
    }
  }
  return V;
}

MatrixXd BernTet(int N, VectorXd r, VectorXd s, VectorXd t)
{

  int Np = (N + 1) * (N + 2) * (N + 3) / 6;
  MatrixXd V(r.rows(), Np);

  VectorXd L1 = -(1 + r.array() + s.array() + t.array()) / 2;
  VectorXd L2 = (1 + r.array()) / 2;
  VectorXd L3 = (1 + s.array()) / 2;
  VectorXd L4 = (1 + t.array()) / 2;

  int sk = 0;
  for (int l = 0; l <= N; ++l)
  {
    for (int k = 0; k <= N - l; ++k)
    {
      for (int j = 0; j <= N - l - k; ++j)
      {
        int i = N - j - k - l;
        double C = (double)factorial(N) /
                   (factorial(i) * factorial(j) * factorial(k) * factorial(l));
        VectorXd L1i = Eigen::pow(L1.array(), i);
        VectorXd L2j = Eigen::pow(L2.array(), j);
        VectorXd L3k = Eigen::pow(L3.array(), k);
        VectorXd L4l = Eigen::pow(L4.array(), l);
        V.col(sk) = C * L1i.array() * L2j.array() * L3k.array() * L4l.array();
        ++sk;
      }
    }
  }

  return V;
}

void GradBernTet(int N, VectorXd r, VectorXd s, VectorXd t, MatrixXd &V1,
                 MatrixXd &V2, MatrixXd &V3, MatrixXd &V4)
{

  int Np = (N + 1) * (N + 2) * (N + 3) / 6;

  VectorXd L1 = -(1 + r.array() + s.array() + t.array()) / 2;
  VectorXd L2 = (1 + r.array()) / 2;
  VectorXd L3 = (1 + s.array()) / 2;
  VectorXd L4 = (1 + t.array()) / 2;

  V1.resize(r.rows(), Np);
  V2.resize(r.rows(), Np);
  V3.resize(r.rows(), Np);
  V4.resize(r.rows(), Np);
  int sk = 0;
  for (int l = 0; l <= N; ++l)
  {
    for (int k = 0; k <= N - l; ++k)
    {
      for (int j = 0; j <= N - l - k; ++j)
      {
        int i = N - j - k - l;
        double C = (double)factorial(N) /
                   (factorial(i) * factorial(j) * factorial(k) * factorial(l));
        VectorXd L1i = Eigen::pow(L1.array(), i);
        VectorXd L2j = Eigen::pow(L2.array(), j);
        VectorXd L3k = Eigen::pow(L3.array(), k);
        VectorXd L4l = Eigen::pow(L4.array(), l);

        VectorXd L1im = Eigen::pow(L1.array(), i - 1);
        VectorXd L2jm = Eigen::pow(L2.array(), j - 1);
        VectorXd L3km = Eigen::pow(L3.array(), k - 1);
        VectorXd L4lm = Eigen::pow(L4.array(), l - 1);

        VectorXd dL1 =
            C * i * L1im.array() * L2j.array() * L3k.array() * L4l.array();
        VectorXd dL2 =
            C * j * L1i.array() * L2jm.array() * L3k.array() * L4l.array();
        VectorXd dL3 =
            C * k * L1i.array() * L2j.array() * L3km.array() * L4l.array();
        VectorXd dL4 =
            C * l * L1i.array() * L2j.array() * L3k.array() * L4lm.array();
        if (i == 0)
          dL1.fill(0.0);
        if (j == 0)
          dL2.fill(0.0);
        if (k == 0)
          dL3.fill(0.0);
        if (l == 0)
          dL4.fill(0.0);

        V1.col(sk) = dL1;
        V2.col(sk) = dL2;
        V3.col(sk) = dL3;
        V4.col(sk) = dL4;

        ++sk;
      }
    }
  }
}

// ===================== geofacs routines =======================

// (v/s)geofacs = array of volume, surface geofacs
MatrixXd vgeofacs3d(VectorXd x, VectorXd y, VectorXd z, MatrixXd Dr,
                    MatrixXd Ds, MatrixXd Dt)
{

  int Npts = (int)Dr.rows();
  MatrixXd vgeofacs(Npts, 10); // rx,sx,tx,ry,sy,ty,rz,sz,tz,J

  VectorXd xr = Dr * x;
  VectorXd xs = Ds * x;
  VectorXd xt = Dt * x;
  VectorXd yr = Dr * y;
  VectorXd ys = Ds * y;
  VectorXd yt = Dt * y;
  VectorXd zr = Dr * z;
  VectorXd zs = Ds * z;
  VectorXd zt = Dt * z;

  VectorXd J =
      xr.array() * (ys.array() * zt.array() - zs.array() * yt.array()) -
      yr.array() * (xs.array() * zt.array() - zs.array() * xt.array()) +
      zr.array() * (xs.array() * yt.array() - ys.array() * xt.array());

  VectorXd rx = (ys.array() * zt.array() - zs.array() * yt.array()) / J.array();
  VectorXd ry =
      -(xs.array() * zt.array() - zs.array() * xt.array()) / J.array();
  VectorXd rz = (xs.array() * yt.array() - ys.array() * xt.array()) / J.array();
  VectorXd sx =
      -(yr.array() * zt.array() - zr.array() * yt.array()) / J.array();
  VectorXd sy = (xr.array() * zt.array() - zr.array() * xt.array()) / J.array();
  VectorXd sz =
      -(xr.array() * yt.array() - yr.array() * xt.array()) / J.array();
  VectorXd tx = (yr.array() * zs.array() - zr.array() * ys.array()) / J.array();
  VectorXd ty =
      -(xr.array() * zs.array() - zr.array() * xs.array()) / J.array();
  VectorXd tz = (xr.array() * ys.array() - yr.array() * xs.array()) / J.array();

  vgeofacs.col(0) = rx;
  vgeofacs.col(1) = sx;
  vgeofacs.col(2) = tx;
  vgeofacs.col(3) = ry;
  vgeofacs.col(4) = sy;
  vgeofacs.col(5) = ty;
  vgeofacs.col(6) = rz;
  vgeofacs.col(7) = sz;
  vgeofacs.col(8) = tz;
  vgeofacs.col(9) = J;
  return vgeofacs;
}

// D(rst)f = matrix mapping nodal values to derivatives at face cubature points
MatrixXd sgeofacs3d(VectorXd x, VectorXd y, VectorXd z, MatrixXd Drf,
                    MatrixXd Dsf, MatrixXd Dtf)
{
  MatrixXd vgeo = vgeofacs3d(x, y, z, Drf, Dsf, Dtf);
  VectorXd rx = vgeo.col(0);
  VectorXd sx = vgeo.col(1);
  VectorXd tx = vgeo.col(2);
  VectorXd ry = vgeo.col(3);
  VectorXd sy = vgeo.col(4);
  VectorXd ty = vgeo.col(5);
  VectorXd rz = vgeo.col(6);
  VectorXd sz = vgeo.col(7);
  VectorXd tz = vgeo.col(8);
  VectorXd J = vgeo.col(9);

  int Npts = (int)Drf.rows();
  MatrixXd sgeofacs(Npts, 4); // nx,ny,nz,sJ

  int Nfpts = Npts / 4; // assume each face has the same quadrature
  VectorXd nx(Npts);
  VectorXd ny(Npts);
  VectorXd nz(Npts);
  for (int i = 0; i < Nfpts; ++i)
  {
    // face 1
    int id = i;
    nx(id) = -tx(id);
    ny(id) = -ty(id);
    nz(id) = -tz(id);

    // face 2
    id += Nfpts;
    nx(id) = -sx(id);
    ny(id) = -sy(id);
    nz(id) = -sz(id);

    // face 3
    id += Nfpts;
    nx(id) = rx(id) + sx(id) + tx(id);
    ny(id) = ry(id) + sy(id) + ty(id);
    nz(id) = rz(id) + sz(id) + tz(id);

    // face 4
    id += Nfpts;
    nx(id) = -rx(id);
    ny(id) = -ry(id);
    nz(id) = -rz(id);
  }
  VectorXd sJ = (nx.array() * nx.array() + ny.array() * ny.array() +
                 nz.array() * nz.array())
                    .array()
                    .sqrt();

  // normalize and scale sJ
  nx.array() /= sJ.array();
  ny.array() /= sJ.array();
  nz.array() /= sJ.array();
  sJ.array() *= J.array();

  sgeofacs.col(0) = nx;
  sgeofacs.col(1) = ny;
  sgeofacs.col(2) = nz;
  sgeofacs.col(3) = sJ;
  return sgeofacs;
}

// ===================== tabulated nodes ========================

/*
  void tet_cubature_duffy(int N, VectorXd &a, VectorXd &wa,
  VectorXd &b,  VectorXd &wb,
  VectorXd &c,  VectorXd &wc){

  int Np = (N+1);
  a.resize(Np); wa.resize(Np);
  b.resize(Np); wb.resize(Np);
  c.resize(Np); wc.resize(Np);

  #define loadnodes_duffy(M)						\
  a(i) = p_r1D_N##M[i]; wa(i) = p_s_N##M[i]; t(i) = p_t_N##M[i];

  }
*/

// requires alpha, beta = integers
void JacobiGQ(int N, int alpha_int, int beta_int, VectorXd &r, VectorXd &w)
{

  double alpha = (double)alpha_int;
  double beta = (double)beta_int;

  int Np = (N + 1);
  r.resize(Np);
  w.resize(Np);

  if (N == 0)
  {
    r(0) = -(alpha + beta) / (alpha + beta + 2.0);
    w(0) = 2.0;
  }
  VectorXd h1(Np);
  for (int i = 0; i < Np; ++i)
  {
    h1(i) = 2 * i + alpha + beta;
  }
  MatrixXd J(Np, Np);
  J.fill(0.0);
  for (int i = 0; i < Np; ++i)
  {
    if (alpha_int == 0 && beta_int == 0)
    {
      J(i, i) = 0.0;
    }
    else
    {
      J(i, i) = -.5 * (alpha * alpha - beta * beta) / (h1(i) + 2.0) / h1(i);
    }
    if (i < N)
    {
      J(i, i + 1) =
          2.0 / (h1(i) + 2.0) *
          sqrt((i + 1.0) * (i + 1.0 + alpha + beta) * (i + 1.0 + alpha) *
               (i + 1.0 + beta) / (h1(i) + 1.0) / (h1(i) + 3.0));
    }
  }
  double tol = 1e-14;
  if (alpha + beta < tol)
  {
    J(0, 0) = 0.0;
  }
  MatrixXd Js = J + J.transpose();

  SelfAdjointEigenSolver<MatrixXd> eig(Js);
  MatrixXd V = eig.eigenvectors();
  r = eig.eigenvalues();

  //  double gamma_alpha = factorial(alpha-1);
  //  double gamma_beta = factorial(alpha-1);

  // assumes alpha,beta = int so gamma(x) = factorial(x-1)
  w = V.row(0).array().square() * pow(2.0, alpha + beta + 1.0) /
      (alpha + beta + 1) * factorial(alpha_int) * factorial(beta_int) /
      factorial(alpha_int + beta_int);

  //  cout << "r = " << r << endl;
  //  cout << "w = " << w << endl;
}

void Nodes2D(int N, VectorXd &r, VectorXd &s)
{
  int Np = (N + 1) * (N + 2) / 2;
  VectorXd rr, ss, tt;
  Nodes3D(N, rr, ss, tt);

  // assume that t = -1 face is first
  r = rr.head(Np);
  s = ss.head(Np);
}

void Nodes3D(int N, VectorXd &r, VectorXd &s, VectorXd &t)
{
  int Np = (N + 1) * (N + 2) * (N + 3) / 6;
  r.resize(Np);
  s.resize(Np);
  t.resize(Np);
#define loadnodes(M)                                                           \
  r(i) = p_r_N##M[i];                                                          \
  s(i) = p_s_N##M[i];                                                          \
  t(i) = p_t_N##M[i];

  for (int i = 0; i < Np; ++i)
  {
    switch (N)
    {
    case 1:
      loadnodes(1);
      break;
    case 2:
      loadnodes(2);
      break;
    case 3:
      loadnodes(3);
      break;
    case 4:
      loadnodes(4);
      break;
    case 5:
      loadnodes(5);
      break;
    case 6:
      loadnodes(6);
      break;
    case 7:
      loadnodes(7);
      break;
    case 8:
      loadnodes(8);
      break;
    case 9:
      loadnodes(9);
      break;
    case 10:
      loadnodes(10);
      break;
    case 11:
      loadnodes(11);
      break;
    case 12:
      loadnodes(12);
      break;
    case 13:
      loadnodes(13);
      break;
    case 14:
      loadnodes(14);
      break;
    case 15:
      loadnodes(15);
      break;
    case 16:
      loadnodes(16);
      break;
    case 17:
      loadnodes(17);
      break;
    case 18:
      loadnodes(18);
      break;
    case 19:
      loadnodes(10);
      break;
    case 20:
      loadnodes(20);
      break;
    }
  }
}

void tet_cubature(int N, VectorXd &rq, VectorXd &sq, VectorXd &tq, VectorXd &wq)
{

#define loadNq(M) Nq = p_Nq_N##M;
#define loadq(M)                                                               \
  rq(i) = p_rq_N##M[i];                                                        \
  sq(i) = p_sq_N##M[i];                                                        \
  tq(i) = p_tq_N##M[i];                                                        \
  wq(i) = p_wq_N##M[i];

  int Nq = 0;
  switch (N)
  {
  case 1:
    loadNq(1);
    break;
  case 2:
    loadNq(2);
    break;
  case 3:
    loadNq(3);
    break;
  case 4:
    loadNq(4);
    break;
  case 5:
    loadNq(5);
    break;
  case 6:
    loadNq(6);
    break;
  case 7:
    loadNq(7);
    break;
  case 8:
    loadNq(8);
    break;
  case 9:
    loadNq(9);
    break;
  case 10:
    loadNq(10);
    break;
  case 11:
    loadNq(11);
    break;
  case 12:
    loadNq(12);
    break;
  case 13:
    loadNq(13);
    break;
  case 14:
    loadNq(14);
    break;
  case 15:
    loadNq(15);
    break;
  case 16:
    loadNq(16);
    break;
  case 17:
    loadNq(17);
    break;
  case 18:
    loadNq(18);
    break;
  case 19:
    loadNq(19);
    break;
  case 20:
    loadNq(20);
    break;
  case 21:
    loadNq(21);
    break;
  }

  rq.resize(Nq);
  sq.resize(Nq);
  tq.resize(Nq);
  wq.resize(Nq);
  for (int i = 0; i < Nq; ++i)
  {
    switch (N)
    {
    case 1:
      loadq(1);
      break;
    case 2:
      loadq(2);
      break;
    case 3:
      loadq(3);
      break;
    case 4:
      loadq(4);
      break;
    case 5:
      loadq(5);
      break;
    case 6:
      loadq(6);
      break;
    case 7:
      loadq(7);
      break;
    case 8:
      loadq(8);
      break;
    case 9:
      loadq(9);
      break;
    case 10:
      loadq(10);
      break;
    case 11:
      loadq(11);
      break;
    case 12:
      loadq(12);
      break;
    case 13:
      loadq(13);
      break;
    case 14:
      loadq(14);
      break;
    case 15:
      loadq(15);
      break;
    case 16:
      loadq(16);
      break;
    case 17:
      loadq(17);
      break;
    case 18:
      loadq(18);
      break;
    case 19:
      loadq(19);
      break;
    case 20:
      loadq(20);
      break;
    case 21:
      loadq(21);
      break;
    }
  }
}

void tri_cubature(int N, VectorXd &rfq, VectorXd &sfq, VectorXd &wfq)
{

#define loadNfq(M) Nfq = p_Nfq_N##M;
#define loadfq(M)                                                              \
  rfq(i) = p_rfq_N##M[i];                                                      \
  sfq(i) = p_sfq_N##M[i];                                                      \
  wfq(i) = p_wfq_N##M[i];

  int Nfq = 0;
  switch (N)
  {
  case 1:
    loadNfq(1);
    break;
  case 2:
    loadNfq(2);
    break;
  case 3:
    loadNfq(3);
    break;
  case 4:
    loadNfq(4);
    break;
  case 5:
    loadNfq(5);
    break;
  case 6:
    loadNfq(6);
    break;
  case 7:
    loadNfq(7);
    break;
  case 8:
    loadNfq(8);
    break;
  case 9:
    loadNfq(9);
    break;
  case 10:
    loadNfq(10);
    break;
  case 11:
    loadNfq(11);
    break;
  case 12:
    loadNfq(12);
    break;
  case 13:
    loadNfq(13);
    break;
  case 14:
    loadNfq(14);
    break;
  case 15:
    loadNfq(15);
    break;
  case 16:
    loadNfq(16);
    break;
  case 17:
    loadNfq(17);
    break;
  case 18:
    loadNfq(18);
    break;
  case 19:
    loadNfq(19);
    break;
  case 20:
    loadNfq(20);
    break;
  case 21:
    loadNfq(21);
    break;
  }

  rfq.resize(Nfq);
  sfq.resize(Nfq);
  wfq.resize(Nfq);
  for (int i = 0; i < Nfq; ++i)
  {
    switch (N)
    {
    case 1:
      loadfq(1);
      break;
    case 2:
      loadfq(2);
      break;
    case 3:
      loadfq(3);
      break;
    case 4:
      loadfq(4);
      break;
    case 5:
      loadfq(5);
      break;
    case 6:
      loadfq(6);
      break;
    case 7:
      loadfq(7);
      break;
    case 8:
      loadfq(8);
      break;
    case 9:
      loadfq(9);
      break;
    case 10:
      loadfq(10);
      break;
    case 11:
      loadfq(11);
      break;
    case 12:
      loadfq(12);
      break;
    case 13:
      loadfq(13);
      break;
    case 14:
      loadfq(14);
      break;
    case 15:
      loadfq(15);
      break;
    case 16:
      loadfq(16);
      break;
    case 17:
      loadfq(17);
      break;
    case 18:
      loadfq(18);
      break;
    case 19:
      loadfq(19);
      break;
    case 20:
      loadfq(20);
      break;
    case 21:
      loadfq(21);
      break;
    }
  }
}

// ===================== lin algo subroutines =====================

VectorXd flatten(MatrixXd &A)
{

  VectorXd a(Map<VectorXd>(A.data(), A.cols() * A.rows()));
  return a;
}

MatrixXd kron(MatrixXd &A, MatrixXd &B)
{
  int ra = (int)A.rows();
  int rb = (int)B.rows();
  int ca = (int)A.cols();
  int cb = (int)B.cols();
  MatrixXd C(ra * rb, ca * cb);
  C.fill(0.0);
  for (int i = 0; i < ra; ++i)
  {
    for (int j = 0; j < ca; ++j)
    {
      C.block(i * rb, j * cb, rb, cb) = A(i, j) * B;
    }
  }
  return C;
}

// backslash
MatrixXd mldivide(MatrixXd &A, MatrixXd &B)
{
  // return A.colPivHouseholderQr().solve(B);
  return A.fullPivHouseholderQr().solve(B);
}

// A/B
MatrixXd mrdivide(MatrixXd &A, MatrixXd &B)
{
  MatrixXd At = A.transpose();
  MatrixXd Bt = B.transpose();
  // return (Bt.colPivHouseholderQr().solve(At)).transpose();
  return (Bt.fullPivHouseholderQr().solve(At)).transpose();
}

// extract sub-vector routine - currently unused
VectorXd extract(const VectorXd &full, const VectorXi &ind)
{
  int num_indices = (int)ind.rows();
  VectorXd target(num_indices);
  for (int i = 0; i < num_indices; i++)
  {
    target(i) = full(ind(i));
  }
  return target;
}

// fixed-bandwidth sparse ids
void get_sparse_ids(MatrixXd A, MatrixXi &cols, MatrixXd &vals)
{
  int nrows = (int)A.rows();
  int ncols = (int)A.cols();

  double maxVal = A.array().abs().maxCoeff();
  double tol = 1e-5;
  MatrixXi boolA = ((A.array().abs()) > tol * maxVal).cast<int>();
  int max_col_vals = (boolA.rowwise().sum()).maxCoeff();
  // printf("max col vals = %d\n",max_col_vals);
  cols.resize(nrows, max_col_vals);
  vals.resize(nrows, max_col_vals);

  cols.fill(0);
  vals.fill(0.0);
  //  cout << boolA << endl;
  //  cout << "max col vals = " << max_col_vals << endl;
  for (int i = 0; i < nrows; ++i)
  {
    int sk = 0;
    for (int j = 0; j < ncols; ++j)
    {
      if (fabs(A(i, j)) > tol * maxVal)
      {
        cols(i, sk) = j;
        vals(i, sk) = A(i, j);
        ++sk;
      }
    }
  }
  //  cout << "vals = " << endl << vals << endl;
}

// ============================= visualization =============================
/*

  void writeVisToGMSH(string fileName, Mesh *mesh, dfloat *Q, int iField, int
  Nfields){

  int timeStep = 0;
  double time = 0.0;
  int K = mesh->K;
  int Dim = 3;
  int N = p_N;

  // make transformation to monomial basis
  MatrixXi monom(p_Np, Dim);
  MatrixXd vdm(p_Np, p_Np);
  for(int i=0, n=0; i<=N; i++){
  for(int j=0; j<=N; j++){
  for(int k=0; k<=N; k++){
  if(i+j+k <= N){
  monom(n,0) = i;
  monom(n,1) = j;
  monom(n,2) = k;
  n++;
  }
  }
  }
  }
  for(int m=0; m<p_Np; m++){
  for(int n=0; n<p_Np; n++){
  double r = mesh->r(n);
  double s = mesh->s(n);
  double t = mesh->t(n);
  vdm(m,n) = pow((r+1)/2.,monom(m,0)) *
  pow((s+1)/2.,monom(m,1)) * pow((t+1)/2.,monom(m,2));
  }
  }
  MatrixXd coeff = vdm.inverse();

  /// write the gmsh file
  ofstream *posFile;
  posFile = new ofstream(fileName.c_str());
  *posFile << "$MeshFormat" << endl;
  *posFile << "2.2 0 8" << endl;
  *posFile << "$EndMeshFormat" << endl;

  /// write the interpolation scheme
  *posFile << "$InterpolationScheme" << endl;
  *posFile << "\"MyInterpScheme\"" << endl;
  *posFile << "1" << endl;
  *posFile << "5 2" << endl;  // 5 2 = tets
  *posFile << p_Np << " " << p_Np << endl;  // size of matrix 'coeff'
  for(int m=0; m<p_Np; m++){
  for(int n=0; n<p_Np; n++)
  *posFile << coeff(m,n) << " ";
  *posFile << endl;
  }
  *posFile << p_Np << " " << Dim << endl;  // size of matrix 'monom'
  for(int n=0; n<p_Np; n++){
  for(int d=0; d<Dim; d++)
  *posFile << monom(n,d) << " ";
  *posFile << endl;
  }
  *posFile << "$EndInterpolationScheme" << endl;

  /// write element node data
  *posFile << "$ElementNodeData" << endl;
  *posFile << "2" << endl;
  *posFile << "\"" << "Field " << iField << "\"" << endl;  /// name of the view
  *posFile << "\"MyInterpScheme\"" << endl;
  *posFile << "1" << endl;
  *posFile << time << endl;
  *posFile << "3" << endl;
  *posFile << timeStep << endl;
  *posFile << "1" << endl;  /// ("numComp")
  *posFile << K << endl;  /// total number of elementNodeData in this file
  for(int k=0; k<K; k++){
  *posFile << mesh->EToGmshE(k) << " " << p_Np;
  for(int i=0; i<p_Np; i++)
  *posFile << " " << Q[i + iField*p_Np + k*p_Np*Nfields];
  *posFile << endl;
  }
  *posFile << "$EndElementNodeData" << endl;

  posFile->close();
  delete posFile;

  }
*/
