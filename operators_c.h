#ifndef OPERATORS_C_H
#define OPERATORS_C_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct
{
  int dim; // dimension

  int N; // polynomial order of elements
  int M; // cubature   order

  int Np; // number of interpolation points
  int Nq; // number of integration   points

  int Nfp; // number of face interpolation points
  int Nfq; // number of face integration   points

  int Nvgeo;
  int Nfgeo;
  int Nfaces;

  dfloat_t *vgeo; // volume  geometric factors
  dfloat_t *fgeo; // face    geometric factors
  dfloat_t *Jq;

  uintloc_t *mapPq; // mapping of face neighbor integration points
  uintloc_t *Fmask; // extract face dofs

  dfloat_t *nrJ, *nsJ, *ntJ; // ref elem normals
  dfloat_t *Drq, *Dsq, *Dtq; // ref elem derivative operators

  dfloat_t *Vq;
  dfloat_t *Pq;

  dfloat_t *VqLq;
  dfloat_t *VqPq;
  dfloat_t *VfPq;
  dfloat_t *Vfqf;
} host_operators_t;

host_operators_t *host_operators_new_2D(int N, int M, uintloc_t E,
                                        uintloc_t *EToE, uint8_t *EToF,
                                        uint8_t *EToO, double *EToVX);

host_operators_t *host_operators_new_3D(int N, int M, uintloc_t E,
                                        uintloc_t *EToE, uint8_t *EToF,
                                        uint8_t *EToO, double *EToVX);

void host_operators_free(host_operators_t *ops);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif
