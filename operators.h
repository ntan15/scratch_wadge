#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wconversion"
#endif
#include <Eigen/Dense>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include <cstdint>
using namespace Eigen;

// testing eigen
void test_eigen();
void test_solve();
void test_basis();

// container for reference arrays (points, weights, interp matrices)
struct ref_elem_data
{
  int N, Nq, Nfaces;
  VectorXd r, s, t;
  VectorXd rfq, sfq, tfq, wfq;
  VectorXd ref_rfq, ref_sfq;
  VectorXd rq, sq, tq, wq;
  VectorXd nrJ, nsJ, ntJ;
  MatrixXd V, Dr, Ds, Dt;
  MatrixXd Vq, Pq, Vfqf, Vfq, Lq;
};

struct geo_elem_data
{
  MatrixXd rxJ, sxJ, txJ, ryJ, syJ, tyJ, rzJ, szJ, tzJ;
  MatrixXd rxJf, sxJf, txJf, ryJf, syJf, tyJf, rzJf, szJf, tzJf;  
  MatrixXd nxJ, nyJ, nzJ, J, sJ;
  MatrixXd xq, yq, zq;
  MatrixXd xf, yf, zf;
};

typedef Matrix<uint32_t, Dynamic, Dynamic> MatrixXu32;
typedef Matrix<uint8_t, Dynamic, Dynamic> MatrixXu8;
typedef Matrix<uint32_t, Dynamic, 1> VectorXu32;
typedef Matrix<uint8_t, Dynamic, 1> VectorXu8;

struct map_elem_data
{
  MatrixXu32 mapPq;
  MatrixXu32 fmask;
};

ref_elem_data *build_ref_ops_2D(int N, int Nq, int Nfq);
geo_elem_data *build_geofacs_2D(ref_elem_data *ref_data,
                                const Ref<MatrixXd> EToVX);
map_elem_data *build_maps_2D(ref_elem_data *ref_data,
                             const Ref<MatrixXu32> EToE,
                             const Ref<MatrixXu8> EToF,
                             const Ref<MatrixXu8> EToO);

ref_elem_data *build_ref_ops_3D(int N, int Nq, int Nfq);
geo_elem_data *build_geofacs_3D(ref_elem_data *ref_data,
                                const Ref<MatrixXd> EToVX);
map_elem_data *build_maps_3D(ref_elem_data *ref_data,
                             const Ref<MatrixXu32> EToE,
                             const Ref<MatrixXu8> EToF,
                             const Ref<MatrixXu8> EToO);

// 1D
void JacobiGQ(int N, int alpha_int, int beta_int, VectorXd &r, VectorXd &w);
VectorXd JacobiP(VectorXd x, double alpha, double beta, int d);
MatrixXd Vandermonde1D(int N, VectorXd r);
VectorXd GradJacobiP(VectorXd x, double alpha, double beta, int p);
MatrixXd Bern1D(int N, VectorXd r);

// 2D
void Nodes2D(int N, VectorXd &r, VectorXd &s);
void rstoab(VectorXd r, VectorXd s, VectorXd &a, VectorXd &b);
VectorXd Simplex2DP(VectorXd a, VectorXd b, int i, int j);
MatrixXd Vandermonde2D(int N, VectorXd r, VectorXd s); // for face
void GradSimplex2DP(VectorXd a, VectorXd b, int id, int jd, VectorXd &V2Dr,
                    VectorXd &V2Ds);
void GradVandermonde2D(int N, VectorXd r, VectorXd s, MatrixXd &V2Dr,
                       MatrixXd &V2Ds);

MatrixXd BernTri(int N, VectorXd r, VectorXd s);

// 3D
void rsttoabc(VectorXd r, VectorXd s, VectorXd t, VectorXd &a, VectorXd &b,
              VectorXd &c);
VectorXd Simplex3DP(VectorXd a, VectorXd b, VectorXd c, int i, int j, int k);
void GradSimplex3DP(VectorXd a, VectorXd b, VectorXd c, int id, int jd, int kd,
                    VectorXd &V3Dr, VectorXd &V3Ds, VectorXd &V3Dt);
MatrixXd Vandermonde3D(int N, VectorXd r, VectorXd s, VectorXd t);
void GradVandermonde3D(int N, VectorXd r, VectorXd s, VectorXd t,
                       MatrixXd &V3Dr, MatrixXd &V3Ds, MatrixXd &V3Dt);

// unused routine - can define WB nodes this way
MatrixXd VandermondeGHsurf(int N, int Npsurf, VectorXd r, VectorXd s,
                           VectorXd t);
// Gordon-Hall blending VDM for a single face
void VandermondeHier(int N, VectorXd r, VectorXd s, VectorXd t, MatrixXd &vertV,
                     MatrixXd &edgeV, MatrixXd &faceV);
void barytors(VectorXd L1, VectorXd L2, VectorXd L3, VectorXd &r, VectorXd &s);

// Bernstein-Bezier basis
MatrixXd BernTet(int N, VectorXd r, VectorXd s, VectorXd t);
void GradBernTet(int N, VectorXd r, VectorXd s, VectorXd t, MatrixXd &V1,
                 MatrixXd &V2, MatrixXd &V3, MatrixXd &V4);

// geofacs
MatrixXd vgeofacs3d(VectorXd x, VectorXd y, VectorXd z, MatrixXd Dr,
                    MatrixXd Ds, MatrixXd Dt);
MatrixXd sgeofacs3d(VectorXd x, VectorXd y, VectorXd z, MatrixXd Drf,
                    MatrixXd Dsf, MatrixXd Dtf);

// extract tabulated nodes
void Nodes3D(int N, VectorXd &r, VectorXd &s, VectorXd &t);
void tet_cubature(int N, VectorXd &rq, VectorXd &sq, VectorXd &tq,
                  VectorXd &wq);
void tri_cubature(int N, VectorXd &rq, VectorXd &sq, VectorXd &wq);
// void tet_cubature_duffy(int N, VectorXd &a, VectorXd &wa,
//			VectorXd &b,  VectorXd &wb,
//			VectorXd &c,  VectorXd &wc);

// helper routines not present in C++ std lib
unsigned int factorial(int n);
unsigned int factorial_ratio(int n1, int n2);
unsigned int nchoosek(int n, int k);

// linalg helper routines (emulate Matlab)
VectorXd flatten(MatrixXd &A);
MatrixXd kron(MatrixXd &A, MatrixXd &B);
MatrixXd mldivide(MatrixXd &A, MatrixXd &B);                 // A\B
MatrixXd mrdivide(MatrixXd &A, MatrixXd &B);                 // A/B
VectorXd extract(const VectorXd &full, const VectorXi &ind); // currently unused
void get_sparse_ids(MatrixXd A, MatrixXi &cols,
                    MatrixXd &vals); // get fixed-bandwidth sparse ids
