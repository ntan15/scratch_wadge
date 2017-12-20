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

static uintloc_t *to_c(MatrixXu32 &m)
{
  uintloc_t *mdata =
      (uintloc_t *)asd_malloc_aligned(sizeof(uintloc_t) * m.size());
  Eigen::Map<Eigen::Matrix<uintloc_t, Eigen::Dynamic, Eigen::Dynamic>>(
      mdata, m.rows(), m.cols()) = m.cast<uintloc_t>();
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

  //printf("Num cubature points Nq = %d\n",ops->Nq);

  ops->Nfp = N + 1;
  ops->Nfq = (int)ref_data->ref_rfq.size();

  ops->Nfaces = ref_data->Nfaces;
  ops->Nvgeo = 4;
  ops->Nfgeo = 3;

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

  const int Nvgeo = ops->Nvgeo;
  const int Nfgeo = ops->Nfgeo;
  const int Nfaces = ref_data->Nfaces;
  const int Nq = ops->Nq;
  const int Nfq = ops->Nfq;

  ops->xyzq =
    (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * ops->Nq * 3 * E);
  ops->vgeo =
    (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * ops->Nq * Nvgeo * E);
  ops->fgeo = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * ops->Nfq *
                                             Nfgeo * Nfaces * E);

  for (uintloc_t e = 0; e < E; ++e)
  {
    for (int n = 0; n < Nq; ++n)
    {
      ops->xyzq[n + 0*Nq + e*Nq*3] = (dfloat_t)geo_data->xq(n,e);
      ops->xyzq[n + 1*Nq + e*Nq*3] = (dfloat_t)geo_data->yq(n,e);
      //ops->xyzq[n + 2*Nq + e*Nq*3] = (dfloat_t)geo_data->zq(n,e);      
      
      ops->vgeo[e * Nq * Nvgeo + 0 * Nq + n] = (dfloat_t)geo_data->rxJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 1 * Nq + n] = (dfloat_t)geo_data->ryJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 2 * Nq + n] = (dfloat_t)geo_data->sxJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 3 * Nq + n] = (dfloat_t)geo_data->syJ(n, e);
    }
  }

  /*
  for (uintloc_t e = 0; e < E; ++e){
    for (int n = 0; n < Nq*Nvgeo; ++n){
      printf("vgeo[%d] = %f\n",n,ops->vgeo[n + e*Nq*Nvgeo]);
    }
  }
  */

  for (uintloc_t e = 0; e < E; ++e)
  {
    for (int n = 0; n < Nfq * Nfaces; ++n)
    {
      ops->fgeo[e * Nfq * Nfaces * Nfgeo + 0 * Nfq * Nfaces + n] =
          (dfloat_t)geo_data->nxJ(n, e);
      ops->fgeo[e * Nfq * Nfaces * Nfgeo + 1 * Nfq * Nfaces + n] =
          (dfloat_t)geo_data->nyJ(n, e);
      ops->fgeo[e * Nfq * Nfaces * Nfgeo + 2 * Nfq * Nfaces + n] =
          (dfloat_t)geo_data->sJ(n, e);
    }
  }

  ops->Jq = to_c(geo_data->J);

  ops->mapPq = to_c(map_data->mapPq);

  // JC: FIX LATER
  ops->mapPq[0] = 7;
  ops->mapPq[1] = 6;
  ops->mapPq[2] = 9;
  ops->mapPq[3] = 8;
  ops->mapPq[6] = 1;
  ops->mapPq[7] = 0;
  ops->mapPq[8] = 3;
  ops->mapPq[9] = 2;

  //  for(int i = 0; i < Nfq*Nfaces*E; ++i){
  //    printf("mapPq(%d) = %d\n",i,ops->mapPq[i]);
  //  }
  ops->Fmask = to_c(map_data->fmask);

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
  ops->Nfq = (int)ref_data->ref_rfq.size();

  ops->Nfaces = ref_data->Nfaces;
  ops->Nvgeo = 9;
  ops->Nfgeo = 4;

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

  const int Nvgeo = ops->Nvgeo;
  const int Nfgeo = ops->Nfgeo;
  const int Nfaces = ref_data->Nfaces;
  const int Nq = ops->Nq;
  const int Nfq = ops->Nfq;
  ops->vgeo =
      (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * ops->Nq * Nvgeo * E);
  ops->fgeo = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * ops->Nfq *
                                             Nfgeo * Nfaces * E);

  for (uintloc_t e = 0; e < E; ++e)
  {
    for (int n = 0; n < Nq; ++n)
    {
      ops->vgeo[e * Nq * Nvgeo + 0 * Nq + n] = (dfloat_t)geo_data->rxJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 1 * Nq + n] = (dfloat_t)geo_data->ryJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 2 * Nq + n] = (dfloat_t)geo_data->rzJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 3 * Nq + n] = (dfloat_t)geo_data->sxJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 4 * Nq + n] = (dfloat_t)geo_data->syJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 5 * Nq + n] = (dfloat_t)geo_data->szJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 6 * Nq + n] = (dfloat_t)geo_data->txJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 7 * Nq + n] = (dfloat_t)geo_data->tyJ(n, e);
      ops->vgeo[e * Nq * Nvgeo + 8 * Nq + n] = (dfloat_t)geo_data->tzJ(n, e);
    }
  }

  for (uintloc_t e = 0; e < E; ++e)
  {
    for (int n = 0; n < Nfq * Nfaces; ++n)
    {
      ops->fgeo[e * Nfq * Nfaces * Nfgeo + 0 * Nfq * Nfaces + n] =
          (dfloat_t)geo_data->nxJ(n, e);
      ops->fgeo[e * Nfq * Nfaces * Nfgeo + 1 * Nfq * Nfaces + n] =
          (dfloat_t)geo_data->nyJ(n, e);
      ops->fgeo[e * Nfq * Nfaces * Nfgeo + 2 * Nfq * Nfaces + n] =
          (dfloat_t)geo_data->nzJ(n, e);
      ops->fgeo[e * Nfq * Nfaces * Nfgeo + 3 * Nfq * Nfaces + n] =
          (dfloat_t)geo_data->sJ(n, e);
    }
  }

  ops->Jq = to_c(geo_data->J);

  ops->mapPq = to_c(map_data->mapPq);
  ops->Fmask = to_c(map_data->fmask);

  delete ref_data;
  delete geo_data;
  delete map_data;

  return ops;
}

void host_operators_free(host_operators_t *ops)
{
  asd_free_aligned(ops->vgeo);
  asd_free_aligned(ops->fgeo);
  asd_free_aligned(ops->Jq);

  asd_free_aligned(ops->mapPq);
  asd_free_aligned(ops->Fmask);

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
