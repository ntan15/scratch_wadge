#ifndef TYPES_H
#define TYPES_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#include "asd.h"

typedef uint32_t uintloc_t;
#define occaUIntloc(x) occaUInt((uintloc_t)(x))
#define occa_uintlong_name "unsigned int"
#define UINTLOC(x) ASD_APPEND(x, u)
#define UINTLOC_MAX UINT32_MAX
#define UINTLOC_MAX_DIGITS INT32_MAX_DIGITS
#define UINTLOC_MPI MPI_UNSIGNED
#define UINTLOC_PRI PRIu32
#define UINTLOC_SCN SCNu32

typedef uint64_t uintglo_t;
#define occaUIntglo(x) occaULong((uintglo_t)(x))
#define occa_uintglo_name "unsigned long"
#define UINTGLO(x) ASD_APPEND(x, ull)
#define UINTGLO_BITS 64
#define UINTGLO_MAX UINT64_MAX
#define UINTGLO_MAX_DIGITS INT64_MAX_DIGITS
#define UINTGLO_MPI MPI_UNSIGNED_LONG_LONG
#define UINTGLO_PRI PRIu64
#define UINTGLO_SCN SCNu64

#ifdef USE_DFLOAT_DOUBLE
typedef double dfloat_t;
#define occaDfloat occaDouble
#define DFLOAT(x) (x)
#define DFLOAT_FMTe "24.16e"
#define DFLOAT_MAX DBL_MAX
#define DFLOAT_MPI MPI_DOUBLE
#define DFLOAT_SQRT sqrt
#define DFLOAT_STRTOD strtod
#else
typedef float dfloat_t;
#define occaDfloat occaFloat
#define DFLOAT(x) ASD_APPEND(x, f)
#define DFLOAT_FMTe "24.16e"
#define DFLOAT_MAX FLT_MAX
#define DFLOAT_MPI MPI_FLOAT
#define DFLOAT_SQRT sqrtf
#define DFLOAT_STRTOD strtof
#endif

#endif
