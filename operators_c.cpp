#include "operators_c.h"
#include "operators.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;

host_operators_t *host_operators_new_2D(int N, int M, uintloc_t E,
                                        uintloc_t *EToE, uint8_t *EToF,
                                        uint8_t *EToO)
{
  host_operators_t *ops =
      (host_operators_t *)asd_malloc(sizeof(host_operators_t));

  return ops;
}

host_operators_t *host_operators_new_3D(int N, int M, uintloc_t E,
                                        uintloc_t *EToE, uint8_t *EToF,
                                        uint8_t *EToO)
{
  host_operators_t *ops =
      (host_operators_t *)asd_malloc(sizeof(host_operators_t));

  return ops;
}

void host_operators_free(host_operators_t *ops)
{
#if 0
  asd_free_aligned(ops->vgeo);
  asd_free_aligned(ops->fgeo);

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
  asd_free_aligned(ops->Jq);
  asd_free_aligned(ops->Pq);

  asd_free_aligned(ops->VqLq);
  asd_free_aligned(ops->VqPq);
  asd_free_aligned(ops->VfPq);
  asd_free_aligned(ops->Vfqf);
#endif
}
