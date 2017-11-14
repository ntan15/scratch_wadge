#include "operators_c.h"
#include "operators.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;

static dfloat_t *to_c(VectorXd &v)
{
  dfloat_t *vdata = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * v.size());
  Eigen::Map<Eigen::VectorXf>(vdata, v.size()) = v.cast<dfloat_t>();
  return vdata;
}

static dfloat_t *to_c(MatrixXd &m)
{
  dfloat_t *mdata = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * m.size());
  Eigen::Map<Eigen::MatrixXf>(mdata, m.rows(), m.cols()) = m.cast<dfloat_t>();
  return mdata;
}

host_operators_t *host_operators_new_2D(int N, int M, uintloc_t E,
                                        uintloc_t *EToE, uint8_t *EToF,
                                        uint8_t *EToO, double *EToVX)
{
  host_operators_t *ops =
      (host_operators_t *)asd_malloc(sizeof(host_operators_t));

  ref_elem_data *ref_data = build_ref_ops_2D(N, M, M);

  VectorXd wq = ref_data->wq;

  // nodal
  MatrixXd Dr = ref_data->Dr;
  MatrixXd Ds = ref_data->Ds;
  MatrixXd Vq = ref_data->Vq;

  VectorXd wfq = ref_data->wfq;
  VectorXd nrJ = ref_data->nrJ;
  VectorXd nsJ = ref_data->nsJ;
  MatrixXd Vfqf = ref_data->Vfqf;
  MatrixXd Vfq = ref_data->Vfq;

  // quadrature
  MatrixXd MM = Vq.transpose() * wq.asDiagonal() * Vq;
  MatrixXd VqW = Vq.transpose() * wq.asDiagonal();
  MatrixXd Pq = mldivide(MM, VqW);

  MatrixXd MMfq = Vfq.transpose() * wfq.asDiagonal();
  MatrixXd Lq = mldivide(MM, MMfq);
  MatrixXd VqLq = Vq * Lq;
  MatrixXd VqPq = Vq * Pq;
  MatrixXd VfPq = Vfq * Pq;
  MatrixXd Drq = Vq * Dr * Pq - .5 * Vq * Lq * nrJ.asDiagonal() * Vfq * Pq;
  MatrixXd Dsq = Vq * Ds * Pq - .5 * Vq * Lq * nsJ.asDiagonal() * Vfq * Pq;

  ops->dim = 2;

  ops->N = N;
  ops->M = M;

  ops->Np = (int)ref_data->r.size();
  ops->Nq = (int)ref_data->rq.size();

  ops->Nfp = N + 1;
  ops->Nfq = (int)ref_data->rfq.size();

  ops->nrJ = to_c(nrJ);
  ops->nsJ = to_c(nsJ);

  ops->Drq = to_c(Drq);
  ops->Dsq = to_c(Dsq);

  ops->Vq = to_c(Vq);
  ops->Pq = to_c(Pq);

  ops->VqLq = to_c(VqLq);
  ops->VqPq = to_c(VqPq);
  ops->VfPq = to_c(VfPq);
  ops->Vfqf = to_c(Vfqf);

  Map<MatrixXd> EToVXmat(EToVX, 2 * 3, E);

  if (sizeof(uintloc_t) != sizeof(uint32_t))
  {
    cerr << "Need to update build maps to support different integer types"
         << endl;
    std::abort();
  }
  Map<MatrixXu32> mapEToE(EToE, 3, E);
  Map<MatrixXu8> mapEToF(EToF, 3, E);
  Map<MatrixXu8> mapEToO(EToO, 3, E);

  geo_elem_data *geo_data = build_geofacs_2D(ref_data, EToVXmat);
  map_elem_data *map_data = build_maps_2D(ref_data, mapEToE, mapEToF, mapEToO);

  cout << "TODO Fill vgeo, fgeo, and Jq" << endl;
  cout << "TODO Fill fmask and mapPq" << endl;

  delete ref_data;
  delete geo_data;
  delete map_data;

  return ops;
}

host_operators_t *host_operators_new_3D(int N, int M, uintloc_t E,
                                        uintloc_t *EToE, uint8_t *EToF,
                                        uint8_t *EToO, double *EToVX)
{
  host_operators_t *ops =
      (host_operators_t *)asd_malloc(sizeof(host_operators_t));

  ref_elem_data *ref_data = build_ref_ops_3D(N, M, M);

  VectorXd wq = ref_data->wq;

  // nodal
  MatrixXd Dr = ref_data->Dr;
  MatrixXd Ds = ref_data->Ds;
  MatrixXd Dt = ref_data->Dt;
  MatrixXd Vq = ref_data->Vq;

  VectorXd wfq = ref_data->wfq;
  VectorXd nrJ = ref_data->nrJ;
  VectorXd nsJ = ref_data->nsJ;
  VectorXd ntJ = ref_data->ntJ;
  MatrixXd Vfqf = ref_data->Vfqf;
  MatrixXd Vfq = ref_data->Vfq;

  // quadrature
  MatrixXd MM = Vq.transpose() * wq.asDiagonal() * Vq;
  MatrixXd VqW = Vq.transpose() * wq.asDiagonal();
  MatrixXd Pq = mldivide(MM, VqW);

  MatrixXd MMfq = Vfq.transpose() * wfq.asDiagonal();
  MatrixXd Lq = mldivide(MM, MMfq);
  MatrixXd VqLq = Vq * Lq;
  MatrixXd VqPq = Vq * Pq;
  MatrixXd VfPq = Vfq * Pq;
  MatrixXd Drq = Vq * Dr * Pq - .5 * Vq * Lq * nrJ.asDiagonal() * Vfq * Pq;
  MatrixXd Dsq = Vq * Ds * Pq - .5 * Vq * Lq * nsJ.asDiagonal() * Vfq * Pq;
  MatrixXd Dtq = Vq * Dt * Pq - .5 * Vq * Lq * ntJ.asDiagonal() * Vfq * Pq;

  ops->dim = 3;

  ops->N = N;
  ops->M = M;

  ops->Np = (int)ref_data->r.size();
  ops->Nq = (int)ref_data->rq.size();

  ops->Nfp = (N + 1) * (N + 2) / 2;
  ops->Nfq = (int)ref_data->rfq.size();

  ops->nrJ = to_c(nrJ);
  ops->nsJ = to_c(nsJ);
  ops->ntJ = to_c(ntJ);

  ops->Drq = to_c(Drq);
  ops->Dsq = to_c(Dsq);
  ops->Dtq = to_c(Dtq);

  ops->Vq = to_c(Vq);
  ops->Pq = to_c(Pq);

  ops->VqLq = to_c(VqLq);
  ops->VqPq = to_c(VqPq);
  ops->VfPq = to_c(VfPq);
  ops->Vfqf = to_c(Vfqf);

  Map<MatrixXd> EToVXmat(EToVX, 3 * 4, E);

  if (sizeof(uintloc_t) != sizeof(uint32_t))
  {
    cerr << "Need to update build maps to support different integer types"
         << endl;
    std::abort();
  }
  Map<MatrixXu32> mapEToE(EToE, 4, E);
  Map<MatrixXu8> mapEToF(EToF, 4, E);
  Map<MatrixXu8> mapEToO(EToO, 4, E);

  geo_elem_data *geo_data = build_geofacs_3D(ref_data, EToVXmat);
  map_elem_data *map_data = build_maps_3D(ref_data, mapEToE, mapEToF, mapEToO);

  cout << "TODO Fill vgeo, fgeo, and Jq" << endl;
  cout << "TODO Fill fmask and mapPq" << endl;

  delete ref_data;
  delete geo_data;
  delete map_data;

  return ops;
}

void host_operators_free(host_operators_t *ops)
{
  // asd_free_aligned(ops->vgeo);
  // asd_free_aligned(ops->fgeo);
  // asd_free_aligned(ops->Jq);

  // asd_free_aligned(ops->mapPq);
  // asd_free_aligned(ops->Fmask);

  asd_free_aligned(ops->nrJ);
  asd_free_aligned(ops->nsJ);

  asd_free_aligned(ops->Drq);
  asd_free_aligned(ops->Dsq);

  if (ops->dim == 3)
  {
    asd_free_aligned(ops->ntJ);
    asd_free_aligned(ops->Dtq);
  }

  asd_free_aligned(ops->Vq);
  asd_free_aligned(ops->Pq);

  asd_free_aligned(ops->VqLq);
  asd_free_aligned(ops->VqPq);
  asd_free_aligned(ops->VfPq);
  asd_free_aligned(ops->Vfqf);
}
