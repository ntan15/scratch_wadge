//
// euler
//

// {{{ Headers
#include <math.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>

#include <occa_c.h>

#include "asd.h"
#include "operators_c.h"
#include "types.h"

#include <time.h>

// }}}

// {{{ Config Macros
#ifndef ELEM_TYPE
// Undefined element type using Tets
#define ELEM_TYPE 1
#endif
// }}}

// {{{ Unit Macros
#define GiB (1024 * 1024 * 1024)
// }}}

#define TESTING
//#define COALESC

// {{{ Number Types
static uintglo_t strtouglo_or_abort(const char *str)
{
  char *end;
  errno = 0;
  uintmax_t u = strtoumax(str, &end, 10);

  // Error check from
  //   https://www.securecoding.cert.org/confluence/display/c/ERR34-C.+Detect+errors+when+converting+a+string+to+a+number
  if (end == str)
    ASD_ABORT("%s: not a decimal number", str);
  else if ('\0' != *end)
    ASD_ABORT("%s: extra characters at end of input: %s", str, end);
  else if (UINTMAX_MAX == u && ERANGE == errno)
    ASD_ABORT("%s out of range of type uintmax_t", str);
  else if (u > UINTGLO_MAX)
    ASD_ABORT("%ju greater than UINTGLO_MAX", u);

  return u;
}

static double strtodouble_or_abort(const char *str)
{
  char *end;
  errno = 0;
  double x = strtod(str, &end);

  if (end == str)
    ASD_ABORT("%s: not a floating point number", str);
  else if ('\0' != *end)
    ASD_ABORT("%s: extra characters at end of input: %s", str, end);
  else if (ERANGE == errno)
    ASD_ABORT("%s out of range of type double", str);

  return x;
}
// }}}

// {{{ OCCA
/* OCCA modes */
#define UNKNOWN 0
#define SERIAL 1
#define OPENMP 2
#define OPENCL 3
#define CUDA 4
const char *const occa_modes[] = {"UNKNOWN", "SERIAL", "OpenMP",
                                  "OpenCL",  "CUDA",   NULL};

/* OCCA language */
#define OKL_LANG (1 << 0)
#define OFL_LANG (1 << 1)
#define NATIVE_LANG (1 << 2)

static int get_occa_mode(const char *info)
{
  int mode = UNKNOWN;
  if (strstr(info, "Serial"))
    mode = SERIAL;
  else if (strstr(info, "OpenMP"))
    mode = OPENMP;
  else if (strstr(info, "OpenCL"))
    mode = OPENCL;
  else if (strstr(info, "CUDA"))
    mode = CUDA;

  return mode;
}

#define DEVICE_MEMFRAC 0.9 // fraction of device memory to use

#if 0
static void device_async_ptr_to_mem(occaMemory dest, void *src, size_t bytes,
                                    size_t offset)
{
  if (bytes > 0)
    occaAsyncCopyPtrToMem(dest, src, bytes, offset);
}

static void device_async_mem_to_ptr(void *dest, occaMemory src, size_t bytes,
                                    size_t offset)
{
  if (bytes > 0)
    occaAsyncCopyMemToPtr(dest, src, bytes, offset);
}
#endif

static occaMemory device_malloc(occaDevice device, size_t bytecount, void *src)
{
  bytecount = ASD_MAX(1, bytecount);
  uintmax_t bytes = occaDeviceBytesAllocated(device);
  uintmax_t total_bytes = occaDeviceMemorySize(device);
  ASD_ABORT_IF(
      (double)(bytes + bytecount) > DEVICE_MEMFRAC * (double)total_bytes,
      "Over memory limit: \n"
      "      current: allocated %ju (%.2f GiB) out of %ju (%.2f GiB)\n"
      "      new val: allocated %ju (%.2f GiB) out of %ju (%.2f GiB)\n"
      "      (fudge factor is: %.2f)",
      bytes, ((double)bytes) / GiB, total_bytes, ((double)total_bytes) / GiB,
      bytes + bytecount, ((double)(bytes + bytecount)) / GiB, total_bytes,
      ((double)total_bytes) / GiB, DEVICE_MEMFRAC);
  if ((double)(bytes + bytecount) > 0.9 * DEVICE_MEMFRAC * (double)total_bytes)
    ASD_WARNING(
        "At 90%% of memory limit: \n"
        "      current: allocated %ju (%.2f GiB) out of %ju (%.2f GiB)\n"
        "      new val: allocated %ju (%.2f GiB) out of %ju (%.2f GiB)\n"
        "      (fudge factor is: %.2f)",
        bytes, ((double)bytes) / GiB, total_bytes, ((double)total_bytes) / GiB,
        bytes + bytecount, ((double)(bytes + bytecount)) / GiB, total_bytes,
        ((double)total_bytes) / GiB, DEVICE_MEMFRAC);

  return occaDeviceMalloc(device, bytecount, src);
}
// }}}

// {{{ Solver Info
#define APP_NAME "euler"

#if ELEM_TYPE == 0 // triangle
#define NFIELDS 4
#elif ELEM_TYPE == 1 // tetrahedron
#define NFIELDS 5
#else
#error "Unknown/undefined element type"
#endif
// }}}

// {{{ Utilities
static void debug(MPI_Comm comm)
{
  int rank;
  ASD_MPI_CHECK(MPI_Comm_rank(comm, &rank));

  /*
   * This snippet of code is used for parallel debugging.
   *
   * You then need to launch a fresh tmux session which is done by just
   * typing tmux at the command prompt. Next launch your code as usual
   *
   *     mpirun -np 4 ./debug_mpi
   *
   * You can run the following tmux commands to join and synchronise the
   * input to the running processes.
   *
   *     join-pane -s 2
   *     join-pane -s 3
   *     join-pane -s 4
   *     setw synchronize-panes
   */
  char command[ASD_BUFSIZ];
  snprintf(command, ASD_BUFSIZ, "tmux new-window -t %d 'lldb -p %d'", rank + 1,
           getpid());
  printf("command: %s\n", command);
  system(command);

  {
    int pause = 1;
    while (pause)
    {
      /* the code will wait in this loop until the debugger is attached */
    }
  }
}

/** Initialize the libraries that we are using
 *
 */
static void init_libs(MPI_Comm comm, int verbosity)
{
  int rank;
  ASD_MPI_CHECK(MPI_Comm_rank(comm, &rank));

  int loglevel = ASD_MAX(ASD_LL_INFO - verbosity, ASD_LL_ALWAYS);
  asd_log_init(rank, stdout, loglevel);

  // add signal handler to get backtrace on abort
  asd_signal_handler_set();
}

static void print_precision()
{
  ASD_ROOT_INFO("");
  ASD_ROOT_INFO("----- Precision ------------------------------------------");
  ASD_ROOT_INFO("compute precision = %d bytes", sizeof(dfloat_t));
  ASD_ROOT_INFO("----------------------------------------------------------");
}
// }}}

// {{{ Linear Partition
static uintmax_t linpart_starting_row(uintmax_t rank, uintmax_t num_procs,
                                      uintmax_t num_rows)
{
  return (rank * num_rows) / num_procs;
}

static uintmax_t linpart_local_num_rows(uintmax_t rank, uintmax_t num_procs,
                                        uintmax_t num_rows)
{
  return linpart_starting_row(rank + 1, num_procs, num_rows) -
         linpart_starting_row(rank, num_procs, num_rows);
}

#if 0
static uintmax_t linpart_ending_row(uintmax_t rank, uintmax_t num_procs,
                                    uintmax_t num_rows)
{
  return linpart_starting_row(rank + 1, num_procs, num_rows) - 1;
}

static uintmax_t linpart_row_owner(uintmax_t row, uintmax_t num_procs,
                                   uintmax_t num_rows)
{
  return (num_procs * (row + 1) - 1) / num_rows;
}
#endif

// }}}

// {{{ Preferences
typedef struct prefs
{
  lua_State *L;

  MPI_Comm comm;
  int size;     // MPI comm size
  int rank;     // MPI comm rank
  int hostrank; // rank of process on a given host

  char *occa_info;
  char *occa_flags;
  int occa_mode;

  int mesh_N; // order of the polynomial basis
  int mesh_M; // order of the polynomials that can be integrated exactly

  int kernel_KblkU;
  int kernel_KblkV;
  int kernel_KblkF;
  int kernel_KblkS;
  int kernel_T;

  dfloat_t physical_gamma;
  dfloat_t FinalTime;
  dfloat_t CFL;
  dfloat_t tau;

  char *mesh_filename;
  int mesh_sfc_partition;

  char *output_datadir; // directory for output data files
  char *output_prefix;  // prefix for output files
} prefs_t;

static prefs_t *prefs_new(const char *filename, MPI_Comm comm)
{
  prefs_t *prefs = asd_malloc(sizeof(prefs_t));

  lua_State *L = luaL_newstate();
  luaL_openlibs(L);

  ASD_ASSERT(lua_gettop(L) == 0);

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  int hostrank = asd_get_host_rank(comm);

  // set constants for config file
  lua_pushnumber(L, (lua_Number)rank);
  lua_setglobal(L, "MPI_RANK");
  lua_pushnumber(L, (lua_Number)size);
  lua_setglobal(L, "MPI_SIZE");
  lua_pushnumber(L, (lua_Number)hostrank);
  lua_setglobal(L, "HOST_RANK");
  lua_pushnumber(L, (lua_Number)ELEM_TYPE);
  lua_setglobal(L, "ELEM_TYPE");

  ASD_ASSERT(lua_gettop(L) == 0);

  // evaluate config file
  if (luaL_loadfile(L, filename) || lua_pcall(L, 0, 0, 0))
    ASD_LERROR("cannot run configuration file: `%s'", lua_tostring(L, -1));

  // occa
  prefs->occa_info = asd_lua_expr_string(L, "app.occa.info", "mode = Serial");

  int have_occa_flags = asd_lua_expr_boolean(L, "app.occa.flags ~= nil", 0);
  prefs->occa_flags =
      (have_occa_flags) ? asd_lua_expr_string(L, "app.occa.flags", "") : NULL;

  prefs->occa_mode = get_occa_mode(prefs->occa_info);

  prefs->mesh_filename =
      asd_lua_expr_string(L, "app.mesh.filename", "mesh.msh");

  prefs->mesh_N = (int)asd_lua_expr_integer(L, "app.mesh.N", 3);
  prefs->mesh_M = (int)asd_lua_expr_integer(L, "app.mesh.M", 2 * prefs->mesh_N);

  prefs->kernel_KblkV = (int)asd_lua_expr_integer(L, "app.kernel.KblkV", 1);
  prefs->kernel_KblkU = (int)asd_lua_expr_integer(L, "app.kernel.KblkU", 1);
  prefs->kernel_KblkS = (int)asd_lua_expr_integer(L, "app.kernel.KblkS", 1);
  prefs->kernel_KblkF = (int)asd_lua_expr_integer(L, "app.kernel.KblkF", 1);

  prefs->physical_gamma =
      (dfloat_t)asd_lua_expr_number(L, "app.physical.gamma", 1.4);
  prefs->FinalTime =
      (dfloat_t)asd_lua_expr_number(L, "app.physical.FinalTime", .1);
  prefs->CFL =
      (dfloat_t)asd_lua_expr_number(L, "app.physical.CFL", .25);
  prefs->tau =
      (dfloat_t)asd_lua_expr_number(L, "app.physical.tau", 1.0);

  prefs->mesh_sfc_partition =
      asd_lua_expr_boolean(L, "app.mesh.sfc_partition", 1);

  // output
  prefs->output_datadir = asd_lua_expr_string(L, "app.output.datadir", ".");
  {
    struct stat sb;
    if (stat(prefs->output_datadir, &sb) != 0 &&
        mkdir(prefs->output_datadir, 0755) != 0 && errno != EEXIST)
      perror("making datadir");
  }

  prefs->output_prefix = asd_lua_expr_string(L, "app.output.prefix", APP_NAME);

  ASD_ASSERT(lua_gettop(L) == 0);

  prefs->comm = comm;
  prefs->size = size;
  prefs->rank = rank;
  prefs->hostrank = hostrank;

  prefs->L = L;

  return prefs;
}

static void prefs_free(prefs_t *prefs)
{
  lua_close(prefs->L);
  asd_free(prefs->occa_info);
  asd_free(prefs->occa_flags);
  asd_free(prefs->mesh_filename);
  asd_free(prefs->output_datadir);
  asd_free(prefs->output_prefix);
}

static void prefs_print(prefs_t *prefs)
{
  ASD_ROOT_INFO("");
  ASD_ROOT_INFO("----- Preferences Read -----------------------------------");
  ASD_ROOT_INFO("  occa_info  = \"%s\"", prefs->occa_info);
  if (prefs->occa_flags)
    ASD_ROOT_INFO("  occa_flags = \"%s\"", prefs->occa_flags);
  ASD_ROOT_INFO("  occa_mode  = \"%s\"", occa_modes[prefs->occa_mode]);
  ASD_ROOT_INFO("");
  ASD_ROOT_INFO("  mesh_filename = \"%s\"", prefs->mesh_filename);
  ASD_ROOT_INFO("  mesh_N        = %d", prefs->mesh_N);
  ASD_ROOT_INFO("  mesh_M        = %d", prefs->mesh_M);
  ASD_ROOT_INFO("");
  ASD_ROOT_INFO("  kernel_KblkV  = %d", prefs->kernel_KblkV);
  ASD_ROOT_INFO("  kernel_KblkU  = %d", prefs->kernel_KblkU);
  ASD_ROOT_INFO("  kernel_KblkS  = %d", prefs->kernel_KblkS);
  ASD_ROOT_INFO("  kernel_KblkF  = %d", prefs->kernel_KblkF);
  ASD_ROOT_INFO("");
  ASD_ROOT_INFO("  physical_gamma = %g", prefs->physical_gamma);
  ASD_ROOT_INFO("");
  ASD_ROOT_INFO("  output_datadir = %s", prefs->output_datadir);
  ASD_ROOT_INFO("  output_prefix  = %s", prefs->output_prefix);
  ASD_ROOT_INFO("----------------------------------------------------------");
}
// }}}

// {{{ Mesh
#if ELEM_TYPE == 0 // triangle
#define VDIM 2
#define NVERTS 3
#define NFACES 3
#define NFACEVERTS 2
#define NFACEORIEN 2
#define MSH_ELEM_TYPE 2
#define MFEM_ELEM_TYPE 2
#define MFEM_FACE_TYPE 1

const uint8_t FToFV[NFACES * NFACEVERTS] = {
    0, 1, // face 0
    1, 2, // face 1
    2, 0  // face 2
};

// Orientations for triangle faces:
//
//      0          1
//   0-----1    1-----0
//
const uint8_t OToFV[NFACEORIEN * NFACEVERTS] = {
    0, 1, // orientation 0
    1, 0, // orientation 1
};

const uint8_t OToFV_inv[NFACEORIEN * NFACEVERTS] = {
    0, 1, // orientation 0
    1, 0, // orientation 1
};

const uint8_t OOToNO[NFACEORIEN][NFACEORIEN] = {
    {0, 1}, //
    {1, 0}  //
};

#elif ELEM_TYPE == 1 // tetrahedron
#define VDIM 3
#define NVERTS 4
#define NFACES 4
#define NFACEVERTS 3
#define NFACEORIEN 6
#define MSH_ELEM_TYPE 4
#define MFEM_ELEM_TYPE 4
#define MFEM_FACE_TYPE 2

const uint8_t FToFV[NFACES * NFACEVERTS] = {
    0, 1, 2, // face 0
    0, 1, 3, // face 1
    1, 2, 3, // face 2
    0, 2, 3  // face 3
};

// Orientations for tetrahedron faces:
//     0         1         2         3         4         5    //
//    /2\       /1\       /0\       /2\       /0\       /1\   //
//   /   \     /   \     /   \     /   \     /   \     /   \  //
//  /0___1\   /2___0\   /1___2\   /1___0\   /2___1\   /0___2\ //
//
const uint8_t OToFV[NFACEORIEN * NFACEVERTS] = {
    0, 1, 2, // orientation 0
    2, 0, 1, // orientation 1
    1, 2, 0, // orientation 2
    1, 0, 2, // orientation 3
    2, 1, 0, // orientation 4
    0, 2, 1, // orientation 5
};

const uint8_t OToFV_inv[NFACEORIEN * NFACEVERTS] = {
    0, 1, 2, // orientation 0
    1, 2, 0, // orientation 1
    2, 0, 1, // orientation 2
    1, 0, 2, // orientation 3
    2, 1, 0, // orientation 4
    0, 2, 1, // orientation 5
};

// What permutation (OOToNO[o0][o1]) needs to be applied to convert orientation
// o0 into orientation o1.
const uint8_t OOToNO[NFACEORIEN][NFACEORIEN] = {
    {0, 1, 2, 3, 4, 5}, //
    {2, 0, 1, 4, 5, 3}, //
    {1, 2, 0, 5, 3, 4}, //
    {3, 4, 5, 0, 1, 2}, //
    {4, 5, 3, 2, 0, 1}, //
    {5, 3, 4, 1, 2, 0}  //
};

#else
#error "Unknown/undefined element type"
#endif

// Note that the mesh duplicates the vertices for each element (like DG dofs).
// This is done to make partitioning the mesh simple.
typedef struct
{
  uintloc_t E;  // number of total elements on this rank
  uintloc_t ER; // number of real  elements on this rank
  uintloc_t EG; // number of ghost elements on this rank

  uintglo_t *EToVG; // element to global vertex numbers
  double *EToVX;    // element to vertex coordinates

  uintloc_t *EToE; // element to neighboring element
  uint8_t *EToF;   // element to neighboring element face
  uint8_t *EToO;   // element to neighboring element orientation

  uintloc_t EI;     // number of interior elements
  uintloc_t *ESetI; // set of interior elements

  uintloc_t EE;     // number of exterior elements
  uintloc_t *ESetE; // set of exterior elements

  uintloc_t ES;     // number of element to send
  uintloc_t *ESetS; // set of elements to send

  uintloc_t *recv_starts; // element index to receive from each rank
  uintloc_t *send_starts; // index into ESetS to send to each rank
} host_mesh_t;

static void host_mesh_free(host_mesh_t *mesh)
{
  asd_free_aligned(mesh->EToVG);
  asd_free_aligned(mesh->EToVX);

  asd_free_aligned(mesh->EToE);
  asd_free_aligned(mesh->EToF);
  asd_free_aligned(mesh->EToO);

  asd_free_aligned(mesh->ESetI);
  asd_free_aligned(mesh->ESetE);
  asd_free_aligned(mesh->ESetS);

  asd_free_aligned(mesh->recv_starts);
  asd_free_aligned(mesh->send_starts);
}

static host_mesh_t *host_mesh_read_msh(const prefs_t *prefs)
{
  ASD_ROOT_INFO("Reading mesh from '%s'", prefs->mesh_filename);

  host_mesh_t *mesh = asd_malloc(sizeof(host_mesh_t));
  asd_dictionary_t periodic_vertices;
  asd_dictionary_init(&periodic_vertices);

  uintglo_t Nvglo = 0, Nperiodic = 0;
  double *VXglo = NULL;
  uintglo_t Eall = 0, Eglo = 0, e = 0;
  uintglo_t *EToVglo = NULL;

  FILE *fid = fopen(prefs->mesh_filename, "rb");
  ASD_ABORT_IF_NOT(fid != NULL, "Failed to open %s", prefs->mesh_filename);

  int reading_nodes = 0, reading_elements = 0, reading_periodic = 0;

  // Currently we are reading the whole mesh into memory and only keeping a part
  // if it.  If this becomes a bottle neck we can thing about other ways of
  // getting a mesh.

  for (;;)
  {
    char *line = asd_getline(fid);

    if (line == NULL)
      break;

    if (line[0] == '$')
    {
      reading_periodic = reading_elements = reading_nodes = 0;

      if (strstr(line, "$Nodes"))
      {
        reading_nodes = 1;
        asd_free(line);
        line = asd_getline(fid);
        Nvglo = strtouglo_or_abort(line);
        VXglo = asd_malloc_aligned(sizeof(double) * VDIM * Nvglo);
      }
      else if (strstr(line, "$Elements"))
      {
        reading_elements = 1;
        asd_free(line);
        line = asd_getline(fid);
        Eall = strtouglo_or_abort(line);
        EToVglo = asd_malloc_aligned(sizeof(uintglo_t) * NVERTS * Eall);
      }
      else if (strstr(line, "$Periodic"))
      {
        reading_periodic = 1;
        asd_free(line);
        line = asd_getline(fid);
        Nperiodic = strtouglo_or_abort(line);
      }

      if (reading_nodes || reading_elements || reading_periodic)
      {
        asd_free(line);
        continue;
      }
    }

    if (reading_nodes)
    {
      char *word = strtok(line, " ");
      uintglo_t v = strtouglo_or_abort(word);
      for (int d = 0; d < VDIM; ++d)
      {
        word = strtok(NULL, " ");
        double x = strtodouble_or_abort(word);
        VXglo[VDIM * (v - 1) + d] = x;
      }
    }
    else if (reading_elements)
    {
      ASD_TRACE("reading_elements from '%s'", line);
      strtok(line, " ");
      char *word = strtok(NULL, " ");
      uintglo_t elemtype = strtouglo_or_abort(word);

      if (MSH_ELEM_TYPE == elemtype)
      {
        ASD_TRACE("Reading tri", line);
        ASD_ABORT_IF_NOT(e < Eall, "Too many elements");
        word = strtok(NULL, " ");
        uintglo_t numtags = strtouglo_or_abort(word);

        ASD_TRACE("Reading %ju tags", numtags);
        // skip tags for now
        for (uintglo_t t = 0; t < numtags; ++t)
          strtok(NULL, " ");

        for (int n = 0; n < NVERTS; ++n)
        {
          word = strtok(NULL, " ");
          uintglo_t v = strtouglo_or_abort(word);
          EToVglo[NVERTS * e + n] = v - 1;
        }
        ++e;
      }
    }
    else if (reading_periodic)
    {
      for (uintglo_t p = 0; p < Nperiodic; ++p)
      {
        asd_free(line); // "dimension child-entity-tag motherr-entity-tag" line

        line = asd_getline(fid);
        if (strstr(line, "Affine"))
        {
          asd_free(line); // "Affine ..." line
          line = asd_getline(fid);
        }

        uintglo_t Npoints = strtouglo_or_abort(line);
        ASD_TRACE("Reading %ju periodic points", Npoints);

        for (uintglo_t n = 0; n < Npoints; ++n)
        {
          asd_free(line);
          line = asd_getline(fid);
// Don't read periodic section
#if 0
          char *child = strtok(line, " ");
          char *mother = strtok(NULL, " ");

          ASD_TRACE("Periodic %s -> %s", child, mother);
          asd_dictionary_insert(&periodic_vertices, child, mother);
#endif
        }
        asd_free(line);
        line = asd_getline(fid);
      }
    }

    asd_free(line);
  }
  fclose(fid);

  Eglo = e;

  ASD_ABORT_IF_NOT(VXglo, "Nodes section not found in %s",
                   prefs->mesh_filename);
  ASD_ABORT_IF_NOT(EToVglo, "Elements section not found in %s",
                   prefs->mesh_filename);

#ifdef ASD_DEBUG
  ASD_TRACE("Dumping global mesh");
  ASD_TRACE("  Num Vertices %ju", (intmax_t)Nvglo);
  for (uintglo_t v = 0; v < Nvglo; ++v)
    ASD_TRACE("%5" UINTGLO_PRI " %24.16e %24.16e", v, VXglo[VDIM * v + 0],
              VXglo[VDIM * v + 1]);

  ASD_TRACE("  Num Elements %ju", (intmax_t)Eglo);
  for (uintglo_t e = 0; e < Eglo; ++e)
    ASD_TRACE("%5" UINTGLO_PRI " %5" UINTGLO_PRI " %5" UINTGLO_PRI
              " %5" UINTGLO_PRI,
              e, EToVglo[NVERTS * e + 0], EToVglo[NVERTS * e + 1],
              EToVglo[NVERTS * e + 2]);
#endif

  // Compute partition
  uintglo_t ethis = linpart_starting_row(prefs->rank, prefs->size, Eglo);
  uintglo_t enext = linpart_starting_row(prefs->rank + 1, prefs->size, Eglo);
  uintglo_t E = enext - ethis;
  ASD_ABORT_IF_NOT(E <= UINTLOC_MAX,
                   "Local number of elements %ju too big for uintloc_t max %ju",
                   (uintmax_t)E, (uintmax_t)UINTLOC_MAX);

  ASD_VERBOSE("Keeping %ju elements on rank %d", (intmax_t)E, prefs->rank);

  mesh->E = (uintloc_t)E;
  mesh->EToVG = asd_malloc_aligned(sizeof(uintglo_t) * NVERTS * E);
  mesh->EToVX = asd_malloc_aligned(sizeof(double) * NVERTS * VDIM * E);

  for (e = ethis; e < enext; ++e)
  {
    for (int n = 0; n < NVERTS; ++n)
    {
      uintglo_t v = EToVglo[NVERTS * e + n];

      // Use the non-periodic vertex numbers to set the geometry
      for (int d = 0; d < VDIM; ++d)
        mesh->EToVX[NVERTS * VDIM * (e - ethis) + VDIM * n + d] =
            VXglo[VDIM * v + d];

      // Store the periodic vertex numbers for the topology (i.e., connectivity)
      char child[ASD_BUFSIZ];
      snprintf(child, ASD_BUFSIZ, "%" UINTGLO_PRI, v + 1);
// Turn off gmsh based periodic crap
#if 0
      char *mother = asd_dictionary_get_value(&periodic_vertices, child);

      if (mother)
        ASD_TRACE("Finding periodic mother vertex for child %ju", v);

      while (mother)
      {
        const uintglo_t mv = strtouglo_or_abort(mother) - 1;
        ASD_TRACE("  %ju -> %ju", v, mv);
        v = mv;

        snprintf(child, ASD_BUFSIZ, "%" UINTGLO_PRI, v + 1);
        mother = asd_dictionary_get_value(&periodic_vertices, child);
      }
#endif

      mesh->EToVG[NVERTS * (e - ethis) + n] = v;
    }

    for (int n0 = 0; n0 < NVERTS; ++n0)
    {
      const uintglo_t v0 = mesh->EToVG[NVERTS * (e - ethis) + n0];

      for (int n1 = 0; n1 < NVERTS; ++n1)
      {
        if (n0 != n1)
        {
          const uintglo_t v1 = mesh->EToVG[NVERTS * (e - ethis) + n1];

          ASD_ABORT_IF_NOT(
              v0 != v1,
              "\n         Sometimes coarse periodic meshes end up with\n"
              "         elements that have the same vertex number.  If\n"
              "         this happens we don't know how to connect the element\n"
              "         because we depend on unique vertex numbers to\n"
              "         connect the faces.  So we just bomb out.  A\n"
              "         workaround is to use a sufficiently refined mesh\n"
              "         that doesn't have elements with the same vertex.");
        }
      }
    }
  }

  asd_free_aligned(VXglo);
  asd_free_aligned(EToVglo);
  asd_dictionary_clear(&periodic_vertices);

  // Initialize an unconnected mesh
  mesh->ER = mesh->E;
  mesh->EG = 0;

  mesh->EToE = asd_malloc_aligned(sizeof(uintloc_t) * NFACES * E);
  mesh->EToF = asd_malloc_aligned(sizeof(uint8_t) * NFACES * E);
  mesh->EToO = asd_malloc_aligned(sizeof(uint8_t) * NFACES * E);

  for (uintloc_t e = 0; e < mesh->E; ++e)
  {
    for (uint8_t f = 0; f < NFACES; ++f)
    {
      mesh->EToE[NFACES * e + f] = e;
      mesh->EToF[NFACES * e + f] = f;
      mesh->EToO[NFACES * e + f] = 0;
    }
  }

  mesh->EI = 0;
  mesh->EE = 0;
  mesh->ES = 0;

  mesh->ESetI = asd_malloc_aligned(sizeof(uintloc_t) * mesh->EI);
  mesh->ESetE = asd_malloc_aligned(sizeof(uintloc_t) * mesh->EE);
  mesh->ESetS = asd_malloc_aligned(sizeof(uintloc_t) * mesh->ES);

  mesh->recv_starts = asd_malloc_aligned(sizeof(uintloc_t) * (prefs->size + 1));
  mesh->send_starts = asd_malloc_aligned(sizeof(uintloc_t) * (prefs->size + 1));

  for (int r = 0; r <= prefs->size; ++r)
  {
    mesh->recv_starts[r] = 0;
    mesh->send_starts[r] = 0;
  }
  mesh->recv_starts[prefs->rank + 1] = mesh->E;

  return mesh;
}

#define FN_R0 (NFACEVERTS + 0)
#define FN_E0 (NFACEVERTS + 1)
#define FN_F0 (NFACEVERTS + 2)
#define FN_O0 (NFACEVERTS + 3)
#define FN_R1 (NFACEVERTS + 4)
#define FN_E1 (NFACEVERTS + 5)
#define FN_F1 (NFACEVERTS + 6)
#define FN_O1 (NFACEVERTS + 7)
#define NFN (NFACEVERTS + 8)

int uintloc_cmp(const void *a, const void *b)
{
  const uintloc_t *ia = (const uintloc_t *)a;
  const uintloc_t *ib = (const uintloc_t *)b;
  int retval = 0;

  if (*ia > *ib)
    retval = 1;
  else if (*ia < *ib)
    retval = -1;

  return retval;
}

int fn_cmp(const void *a, const void *b)
{
  const uintglo_t *ia = (const uintglo_t *)a;
  const uintglo_t *ib = (const uintglo_t *)b;
  int retval = 0;

  for (int v = 0; v < NFACEVERTS; ++v)
  {
    if (ia[v] > ib[v])
      retval = 1;
    else if (ia[v] < ib[v])
      retval = -1;
    if (retval != 0)
      break;
  }

  return retval;
}

int fn_r0r1e0f0_cmp(const void *a, const void *b)
{
  const uintglo_t *ia = (const uintglo_t *)a;
  const uintglo_t *ib = (const uintglo_t *)b;
  int retval = 0;

  if (ia[FN_R0] > ib[FN_R0])
    retval = 1;
  else if (ia[FN_R0] < ib[FN_R0])
    retval = -1;
  else if (ia[FN_R1] > ib[FN_R1])
    retval = 1;
  else if (ia[FN_R1] < ib[FN_R1])
    retval = -1;
  else if (ia[FN_E0] > ib[FN_E0])
    retval = 1;
  else if (ia[FN_E0] < ib[FN_E0])
    retval = -1;
  else if (ia[FN_F0] > ib[FN_F0])
    retval = 1;
  else if (ia[FN_F0] < ib[FN_F0])
    retval = -1;

  return retval;
}

int fn_r0r1e1f1_cmp(const void *a, const void *b)
{
  const uintglo_t *ia = (const uintglo_t *)a;
  const uintglo_t *ib = (const uintglo_t *)b;
  int retval = 0;

  if (ia[FN_R0] > ib[FN_R0])
    retval = 1;
  else if (ia[FN_R0] < ib[FN_R0])
    retval = -1;
  else if (ia[FN_R1] > ib[FN_R1])
    retval = 1;
  else if (ia[FN_R1] < ib[FN_R1])
    retval = -1;
  else if (ia[FN_E1] > ib[FN_E1])
    retval = 1;
  else if (ia[FN_E1] < ib[FN_E1])
    retval = -1;
  else if (ia[FN_F1] > ib[FN_F1])
    retval = 1;
  else if (ia[FN_F1] < ib[FN_F1])
    retval = -1;

  return retval;
}

// https://en.wikipedia.org/wiki/XOR_swap_algorithm
#define XOR_SWAP(a, b)                                                         \
  do                                                                           \
  {                                                                            \
    (a) ^= (b);                                                                \
    (b) ^= (a);                                                                \
    (a) ^= (b);                                                                \
  } while (0)

static host_mesh_t *host_mesh_connect(MPI_Comm comm, const host_mesh_t *om)
{
  int rank, size;
  ASD_MPI_CHECK(MPI_Comm_rank(comm, &rank));
  ASD_MPI_CHECK(MPI_Comm_size(comm, &size));

  MPI_Request *recv_requests =
      asd_malloc_aligned(sizeof(MPI_Request) * 2 * size);
  MPI_Request *send_requests =
      asd_malloc_aligned(sizeof(MPI_Request) * 2 * size);

  host_mesh_t *nm = asd_malloc(sizeof(host_mesh_t));

  uintglo_t *fnloc =
      asd_malloc_aligned(sizeof(uintglo_t) * NFN * NFACES * om->E);

  // {{{ fill fnloc
  for (uintloc_t e = 0, fn = 0; e < om->E; ++e)
  {
    for (uint8_t f = 0; f < NFACES; ++f, ++fn)
    {
      uintglo_t fv[NFACEVERTS], o = 0;
      for (uint8_t v = 0; v < NFACEVERTS; ++v)
        fv[v] = om->EToVG[NVERTS * e + FToFV[NFACEVERTS * f + v]];

// Use a sorting network from <http://pages.ripco.net/~jgamble/nw.html> to sort
// the vertices.  Also compute orientation relative to the sorted order.
#if NFACEVERTS == 2
      {
        if (fv[1] < fv[0])
        {
          XOR_SWAP(fv[0], fv[1]);
          o = 1;
        }
      }
#elif NFACEVERTS == 3
      {
        int s0 = 0, s1 = 0, s2 = 0;

        if (fv[2] < fv[1])
        {
          XOR_SWAP(fv[1], fv[2]);
          s0 = 1;
        }

        if (fv[2] < fv[0])
        {
          XOR_SWAP(fv[0], fv[2]);
          s1 = 1;
        }

        if (fv[1] < fv[0])
        {
          XOR_SWAP(fv[0], fv[1]);
          s2 = 1;
        }

        if (s0 == 0 && s1 == 0 && s2 == 0)
          o = 0;
        else if (s0 == 0 && s1 == 1 && s2 == 1)
          o = 1;
        else if (s0 == 1 && s1 == 0 && s2 == 1)
          o = 2;
        else if (s0 == 0 && s1 == 0 && s2 == 1)
          o = 3;
        else if (s0 == 1 && s1 == 1 && s2 == 1)
          o = 4;
        else if (s0 == 1 && s1 == 0 && s2 == 0)
          o = 5;
        else
          ASD_ABORT("This should never be reached; problem sorting tet face\n"
                    "fv = %3ju %3ju %3ju\n"
                    "s = %d %d %d\n",
                    (uintmax_t)fv[0], (uintmax_t)fv[1], (uintmax_t)fv[2], s0,
                    s1, s2);
      }

#else
#error "sorting network for vertices not defined"
#endif

#if ASD_DEBUG
      for (uint8_t v = 0; v < NFACEVERTS; ++v)
        ASD_ABORT_IF_NOT(fv[OToFV[NFACEVERTS * o + v]] ==
                             om->EToVG[NVERTS * e + FToFV[NFACEVERTS * f + v]],
                         "Problem with orientation and sorted to vertex map");

      for (uint8_t v = 0; v < NFACEVERTS; ++v)
        ASD_ABORT_IF_NOT(
            fv[v] ==
                om->EToVG[NVERTS * e + FToFV[NFACEVERTS * f +
                                             OToFV_inv[NFACEVERTS * o + v]]],
            "Problem with orientation and sorted to vertex inverse map");
#endif

      for (uint8_t v = 0; v < NFACEVERTS; ++v)
        fnloc[NFN * fn + v] = fv[v];

      fnloc[NFN * fn + FN_R0] = rank;
      fnloc[NFN * fn + FN_E0] = e;
      fnloc[NFN * fn + FN_F0] = f;
      fnloc[NFN * fn + FN_O0] = o;

      fnloc[NFN * fn + FN_R1] = UINTGLO_MAX;
      fnloc[NFN * fn + FN_E1] = UINTGLO_MAX;
      fnloc[NFN * fn + FN_F1] = UINTGLO_MAX;
      fnloc[NFN * fn + FN_O1] = UINTGLO_MAX;
    }
  }
  // }}}

  qsort(fnloc, NFACES * om->E, sizeof(uintglo_t) * NFN, fn_cmp);

#if 0
  printf("fnloc:\n");
  for (uintloc_t fn = 0; fn < NFACES * om->E; ++fn)
  {
    printf("%3ju ", (uintmax_t)fn);
    for (int n = 0; n < NFN; ++n)
      printf(" %20" UINTGLO_PRI, fnloc[fn * NFN + n]);
    printf("\n");
  }
#endif

  // {{{ select pivots
  uintglo_t *pivotsloc, *pivotsglo;

  pivotsloc = asd_malloc_aligned(sizeof(uintglo_t) * NFN * size);
  pivotsglo = (rank == 0)
                  ? asd_malloc_aligned(sizeof(uintglo_t) * NFN * size * size)
                  : NULL;

  for (int r = 0; r < size; ++r)
    for (int d = 0; d < NFN; ++d)
      pivotsloc[r * NFN + d] = fnloc[((NFACES * om->E * r) / size) * NFN + d];

#if 0
  printf("pivotsloc\n");
  for (int r = 0; r < size; ++r)
  {
    printf("%2d ", r);
    for (int d = 0; d < NFN; ++d)
      printf(" %20" UINTGLO_PRI, pivotsloc[r * NFN + d]);
    printf("\n");
  }
#endif

  ASD_MPI_CHECK(MPI_Gather(pivotsloc, size * NFN, UINTGLO_MPI, pivotsglo,
                           size * NFN, UINTGLO_MPI, 0, comm));

#if 0
  if (rank == 0)
  {
    printf("pivotsglo\n");
    for (int r = 0; r < size; ++r)
      for (int s = 0; s < size; ++s)
      {
        printf("%2d %2d ", r, s);
        for (int d = 0; d < NFN; ++d)
          printf(" %20" UINTGLO_PRI, pivotsglo[r * size * NFN + s * NFN + d]);
        printf("\n");
      }
  }
#endif

  if (rank == 0)
  {
    uintglo_t *sorted_pivotsglo =
        asd_malloc_aligned(sizeof(uintglo_t) * NFN * size * size);
    asd_multimergesort(sorted_pivotsglo, pivotsglo, size, size,
                       sizeof(uintglo_t) * NFN, fn_cmp);

#if 0
    printf("sorted_pivotsglo\n");
    for (int r = 0; r < size; ++r)
      for (int s = 0; s < size; ++s)
      {
        printf("%2d %2d ", r, s);
        for (int d = 0; d < NFN; ++d)
          printf(" %20" UINTGLO_PRI,
                 sorted_pivotsglo[r * size * NFN + s * NFN + d]);
        printf("\n");
      }
#endif

    for (int r = 0; r < size; ++r)
      for (int d = 0; d < NFN; ++d)
        pivotsloc[r * NFN + d] =
            sorted_pivotsglo[((size * size * r) / size) * NFN + d];

    asd_free_aligned(pivotsglo);
    asd_free_aligned(sorted_pivotsglo);
  }

  ASD_MPI_CHECK(MPI_Bcast(pivotsloc, size * NFN, UINTGLO_MPI, 0, comm));
  // }}}

  // {{{ compute communication map to globally sort fnloc
  uintloc_t *startsloc = asd_calloc(sizeof(uintloc_t), size + 1);
  startsloc[size] = NFACES * om->E;

  // binary search for the starts of each rank in fnloc
  for (int r = 0; r < size; ++r)
  {
    uintloc_t start = 0;
    uintloc_t end = NFACES * om->E - 1;
    uintloc_t offset = 0;

    if (om->E > 0)
    {
      while (end >= start)
      {
        offset = (start + end) / 2;

        if (offset == 0)
          break;

        int c = fn_cmp(fnloc + offset * NFN, pivotsloc + r * NFN);

        if (start == end)
        {
          if (c < 0)
            ++offset;
          break;
        }

        if (c < 0)
          start = offset + 1;
        else if (c > 0)
          end = offset - 1;
        else
          break;
      }

      // Make sure matching faces end up on the same rank
      while (offset > 0 &&
             0 == fn_cmp(fnloc + (offset - 1) * NFN, pivotsloc + r * NFN))
        --offset;
    }

    ASD_ABORT_IF_NOT(offset <= NFACES * om->E, "Problem with binary search");

    startsloc[r] = offset;
  }
  asd_free_aligned(pivotsloc);

#if 0
  printf("startsloc\n");
  for (int r = 0; r <= size; ++r)
  {
    printf("%2d %20" UINTLOC_PRI, r, startsloc[r]);
    if (startsloc[r] < NFACES * om->E)
      for (int d = 0; d < NFN; ++d)
        printf(" %20" UINTGLO_PRI, fnloc[startsloc[r] * NFN + d]);
    printf("\n");
  }
  printf("\n\n");
#endif

  // get number of elements to receive
  int *countsglo = asd_malloc(sizeof(int) * size);
  for (int r = 0; r < size; ++r)
  {
    uintmax_t uic = startsloc[r + 1] - startsloc[r];
    ASD_ABORT_IF_NOT(uic < INT_MAX, "Sending more than INT_MAX elements");
    int c = (int)uic;
    ASD_MPI_CHECK(MPI_Gather(&c, 1, MPI_INT, countsglo, 1, MPI_INT, r, comm));
  }
  uintloc_t *startsglo = asd_calloc(sizeof(uintloc_t), size + 1);
  for (int r = 0; r < size; ++r)
    startsglo[r + 1] = startsglo[r] + countsglo[r];
  asd_free(countsglo);

#if 0
  printf("countsglo\n");
  for (int r = 0; r < size; ++r)
    printf("%2d %20ju\n", r, (uintmax_t)(startsglo[r + 1] - startsglo[r]));

  printf("countsloc\n");
  for (int r = 0; r < size; ++r)
    printf("%2d %20ju\n", r, (uintmax_t)(startsloc[r + 1] - startsloc[r]));
#endif
  // }}}

  // {{{ receive fnglo for global sort
  uintglo_t *fnglo =
      asd_malloc_aligned(sizeof(uintglo_t) * NFN * startsglo[size]);

  for (int r = 0; r < size; ++r)
    MPI_Irecv(fnglo + NFN * startsglo[r],
              NFN * (startsglo[r + 1] - startsglo[r]), UINTGLO_MPI, r, 333,
              comm, recv_requests + r);

  for (int r = 0; r < size; ++r)
    MPI_Isend(fnloc + NFN * startsloc[r],
              NFN * (startsloc[r + 1] - startsloc[r]), UINTGLO_MPI, r, 333,
              comm, send_requests + r);

  MPI_Waitall(size, recv_requests, MPI_STATUSES_IGNORE);
  MPI_Waitall(size, send_requests, MPI_STATUSES_IGNORE);

#if 0
  printf("fnglo received:\n");
  for (uintloc_t fn = 0; fn < startsglo[size]; ++fn)
  {
    for (int n = 0; n < NFN; ++n)
      printf(" %20" UINTGLO_PRI, fnglo[fn * NFN + n]);
    printf("\n");
  }
#endif
  // }}}

  // TODO Should we replace with multi-mergesort?
  qsort(fnglo, startsglo[size], sizeof(uintglo_t) * NFN, fn_cmp);

  // {{{ find neighbors

  for (uintloc_t fn = 0; fn < startsglo[size];)
  {
    if (fn + 1 < startsglo[size] &&
        fn_cmp(&fnglo[NFN * fn], &fnglo[NFN * (fn + 1)]) == 0)
    {
#if 0
      ASD_VERBOSE("Matching face %ju %ju %ju %ju %ju %ju",
                  fnglo[NFN * fn + FN_R0], fnglo[NFN * (fn + 1) + FN_R0],
                  fnglo[NFN * fn + FN_E0], fnglo[NFN * (fn + 1) + FN_E0],
                  fnglo[NFN * fn + FN_F0], fnglo[NFN * (fn + 1) + FN_F0],
                  fnglo[NFN * fn + FN_O0], fnglo[NFN * (fn + 1) + FN_O0]);
#endif

      // found face between elements
      fnglo[NFN * fn + FN_R1] = fnglo[NFN * (fn + 1) + FN_R0];
      fnglo[NFN * fn + FN_E1] = fnglo[NFN * (fn + 1) + FN_E0];
      fnglo[NFN * fn + FN_F1] = fnglo[NFN * (fn + 1) + FN_F0];
      fnglo[NFN * fn + FN_O1] = fnglo[NFN * (fn + 1) + FN_O0];

      fnglo[NFN * (fn + 1) + FN_R1] = fnglo[NFN * fn + FN_R0];
      fnglo[NFN * (fn + 1) + FN_E1] = fnglo[NFN * fn + FN_E0];
      fnglo[NFN * (fn + 1) + FN_F1] = fnglo[NFN * fn + FN_F0];
      fnglo[NFN * (fn + 1) + FN_O1] = fnglo[NFN * fn + FN_O0];

      fn += 2;
    }
    else
    {
      // found unconnected face
      fnglo[NFN * fn + FN_R1] = fnglo[NFN * fn + FN_R0];
      fnglo[NFN * fn + FN_E1] = fnglo[NFN * fn + FN_E0];
      fnglo[NFN * fn + FN_F1] = fnglo[NFN * fn + FN_F0];
      fnglo[NFN * fn + FN_O1] = fnglo[NFN * fn + FN_O0];
      fn += 1;
    }
  }
// }}}

#if 0
  printf("fnglo sorted:\n");
  for (uintloc_t fn = 0; fn < startsglo[size]; ++fn)
  {
    for (int n = 0; n < NFN; ++n)
      printf(" %20" UINTGLO_PRI, fnglo[fn * NFN + n]);
    printf("\n");
  }
#endif

  qsort(fnglo, startsglo[size], sizeof(uintglo_t) * NFN, fn_r0r1e0f0_cmp);

#if 0
  printf("fnglo sorted to send:\n");
  for (uintloc_t fn = 0; fn < startsglo[size]; ++fn)
  {
    for (int n = 0; n < NFN; ++n)
      printf(" %20" UINTGLO_PRI, fnglo[fn * NFN + n]);
    printf("\n");
  }
#endif

  // {{{ receive EToH with partition information
  for (int r = 0; r < size; ++r)
    MPI_Irecv(fnloc + NFN * startsloc[r],
              NFN * (startsloc[r + 1] - startsloc[r]), UINTGLO_MPI, r, 333,
              comm, recv_requests + r);

  for (int r = 0; r < size; ++r)
    MPI_Isend(fnglo + NFN * startsglo[r],
              NFN * (startsglo[r + 1] - startsglo[r]), UINTGLO_MPI, r, 333,
              comm, send_requests + r);

  MPI_Waitall(size, recv_requests, MPI_STATUSES_IGNORE);
  MPI_Waitall(size, send_requests, MPI_STATUSES_IGNORE);

// }}}

#if 0
  printf("fnloc:\n");
  for (uintloc_t fn = 0; fn < NFACES * om->E; ++fn)
  {
    for (int n = 0; n < NFN; ++n)
      printf(" %5" UINTGLO_PRI, fnloc[fn * NFN + n]);
    printf("\n");
  }
#endif

  // {{{ Fill send information
  qsort(fnloc, NFACES * om->E, sizeof(uintglo_t) * NFN, fn_r0r1e0f0_cmp);

  {
    // Count unique sending elements
    uintloc_t ES = 0;
    uintglo_t pr, pe;

    if (om->E > 0)
    {
      pr = fnloc[FN_R1];
      pe = fnloc[FN_E0];

      if (pr != (uintglo_t)rank)
        ++ES;
    }
    for (uintloc_t fn = 1; fn < NFACES * om->E; ++fn)
    {
      const uintglo_t nr = fnloc[NFN * fn + FN_R1];
      const uintglo_t ne = fnloc[NFN * fn + FN_E0];

      if (nr != (uintglo_t)rank && !(pr == nr && pe == ne))
      {
        ++ES;
        pr = nr;
        pe = ne;
      }
    }
    nm->ES = ES;
  }

  {
    // Collect sending map
    uintloc_t es = 0;
    uintglo_t pr, pe;

    uintloc_t *ESetS = asd_malloc_aligned(sizeof(uintloc_t) * nm->ES);
    uintloc_t *send_starts = asd_malloc_aligned(sizeof(uintloc_t) * (size + 1));

    for (int r = 0; r <= size; ++r)
      send_starts[r] = 0;

    if (om->E > 0)
    {
      pr = fnloc[FN_R1];
      pe = fnloc[FN_E0];

      if (pr != (uintglo_t)rank)
      {
        ASD_ABORT_IF_NOT(pe <= UINTLOC_MAX, "Bad element size");
        ESetS[es] = (uintloc_t)pe;
        ++send_starts[pr + 1];
        ++es;
      }
    }
    for (uintloc_t fn = 1; fn < NFACES * om->E; ++fn)
    {
      const uintglo_t nr = fnloc[NFN * fn + FN_R1];
      const uintglo_t ne = fnloc[NFN * fn + FN_E0];

      if (nr != (uintglo_t)rank && !(pr == nr && pe == ne))
      {
        ASD_ABORT_IF_NOT(ne <= UINTLOC_MAX, "Bad element size");
        ESetS[es] = (uintloc_t)ne;
        ++es;
        ++send_starts[nr + 1];
        pr = nr;
        pe = ne;
      }
    }
    for (int r = 0; r < size; ++r)
      send_starts[r + 1] += send_starts[r];

    ASD_ASSERT(send_starts[size] == nm->ES);

    nm->ESetS = ESetS;
    nm->send_starts = send_starts;
  }
  // }}}

  // {{{ Fill recv information and ghost/body information
  qsort(fnloc, NFACES * om->E, sizeof(uintglo_t) * NFN, fn_r0r1e1f1_cmp);

#if 0
  printf("fnloc before recv info:\n");
  for (uintloc_t fn = 0; fn < NFACES * om->E; ++fn)
  {
    for (int n = 0; n < NFN; ++n)
      printf(" %5" UINTGLO_PRI, fnloc[fn * NFN + n]);
    printf("\n");
  }
#endif

  {
    uintloc_t er = 0;
    uintglo_t pr, pe;
    const uintloc_t ER = om->E;

    uintloc_t *recv_starts = asd_malloc_aligned(sizeof(uintloc_t) * (size + 1));

    for (int r = 0; r <= size; ++r)
      recv_starts[r] = 0;

    if (om->E > 0)
    {
      pr = fnloc[FN_R1];
      pe = fnloc[FN_E1];

      if (pr != (uintglo_t)rank)
      {
        ASD_ABORT_IF_NOT(pe <= UINTLOC_MAX, "Bad element size");
        ++recv_starts[pr + 1];

        fnloc[FN_R1] = rank;
        fnloc[FN_E1] = ER + er;

        ++er;
      }
    }
    for (uintloc_t fn = 1; fn < NFACES * om->E; ++fn)
    {
      const uintglo_t nr = fnloc[NFN * fn + FN_R1];
      const uintglo_t ne = fnloc[NFN * fn + FN_E1];

      if (nr != (uintglo_t)rank)
      {
        fnloc[NFN * fn + FN_R1] = rank;

        if (!(pr == nr && pe == ne))
        {
          ASD_ABORT_IF_NOT(ne <= UINTLOC_MAX, "Bad element size");

          fnloc[NFN * fn + FN_E1] = ER + er;

          ++er;
          ++recv_starts[nr + 1];
          pr = nr;
          pe = ne;
        }
        else if (er > 0)
        {
          fnloc[NFN * fn + FN_E1] = ER + er - 1;
        }
      }
    }
    for (int r = 0; r < size; ++r)
      recv_starts[r + 1] += recv_starts[r];

    nm->recv_starts = recv_starts;

    nm->ER = ER;
    nm->EG = recv_starts[size];
    nm->E = nm->ER + nm->EG;

#if 0
    printf("fnloc after recv info:\n");
    for (uintloc_t fn = 0; fn < NFACES * om->E; ++fn)
    {
      for (int n = 0; n < NFN; ++n)
        printf(" %5" UINTGLO_PRI, fnloc[fn * NFN + n]);
      printf("\n");
    }
#endif
  }
  // }}}

  // {{{ Find exterior elements
  {
    uintloc_t EE = 0;

    uintloc_t *ESetE = asd_malloc_aligned(sizeof(uintloc_t) * nm->ES);
    memcpy(ESetE, nm->ESetS, sizeof(uintloc_t) * nm->ES);
    qsort(ESetE, nm->ES, sizeof(uintloc_t), uintloc_cmp);

    if (nm->ES > 0)
    {
      uintloc_t pe = ESetE[EE++];

      for (uintloc_t es = 1; es < nm->ES; ++es)
        if (pe != ESetE[es])
          pe = ESetE[EE++] = ESetE[es];
    }

    nm->EE = EE;
    nm->ESetE = ESetE;
  }
  // }}}

  // {{{ Find interior elements
  {
    uintloc_t EI = 0;

    uintloc_t *ESetI = asd_malloc_aligned(sizeof(uintloc_t) * om->E);

    for (uintloc_t e = 0; e < om->E; ++e)
      ESetI[e] = e;

    for (uintloc_t ee = 0; ee < nm->EE; ++ee)
      ESetI[ee] = UINTLOC_MAX;

    qsort(ESetI, om->E, sizeof(uintloc_t), uintloc_cmp);

    for (uintloc_t e = 0; e < om->E; ++e)
    {
      if (ESetI[e] == UINTLOC_MAX)
      {
        EI = e;
        break;
      }
    }

    nm->EI = EI;
    nm->ESetI = ESetI;
  }
  // }}}

  // {{{ Fill EToE, EToF, and EToO
  {
    uintloc_t *EToE = asd_malloc_aligned(sizeof(uintloc_t) * NFACES * nm->E);
    uint8_t *EToF = asd_malloc_aligned(sizeof(uint8_t) * NFACES * nm->E);
    uint8_t *EToO = asd_malloc_aligned(sizeof(uint8_t) * NFACES * nm->E);

    // Default all elements to unconnected
    for (uintloc_t e = 0; e < nm->E; ++e)
    {
      for (uint8_t f = 0; f < NFACES; ++f)
      {
        EToE[NFACES * e + f] = e;
        EToF[NFACES * e + f] = f;
        EToO[NFACES * e + f] = 0;
      }
    }

    for (uintloc_t fn = 0; fn < NFACES * om->E; ++fn)
    {
      const uintglo_t r0 = fnloc[NFN * fn + FN_R0];
      const uintglo_t e0 = fnloc[NFN * fn + FN_E0];
      const uintglo_t f0 = fnloc[NFN * fn + FN_F0];
      const uintglo_t o0 = fnloc[NFN * fn + FN_O0];
      const uintglo_t r1 = fnloc[NFN * fn + FN_R1];
      const uintglo_t e1 = fnloc[NFN * fn + FN_E1];
      const uintglo_t f1 = fnloc[NFN * fn + FN_F1];
      const uintglo_t o1 = fnloc[NFN * fn + FN_O1];

      ASD_ABORT_IF_NOT(r0 == (uintglo_t)rank,
                       "Problem with local element rank");
      ASD_ABORT_IF_NOT(r1 == (uintglo_t)rank,
                       "Problem with neigh element rank");

      ASD_ABORT_IF_NOT(e0 <= UINTLOC_MAX, "Problem with element size");
      ASD_ABORT_IF_NOT(f0 <= UINT8_MAX, "Problem with face size");
      ASD_ABORT_IF_NOT(e1 <= UINTLOC_MAX, "Problem with element size");
      ASD_ABORT_IF_NOT(f1 <= UINT8_MAX, "Problem with face size");

      EToE[NFACES * e0 + f0] = (uintloc_t)e1;
      EToF[NFACES * e0 + f0] = (uint8_t)f1;
      EToO[NFACES * e0 + f0] = OOToNO[o0][o1];
    }

    nm->EToE = EToE;
    nm->EToF = EToF;
    nm->EToO = EToO;
  }
  // }}}

  // {{{ Fill EToVG and EToVX
  {
    nm->EToVX = asd_malloc_aligned(sizeof(double) * VDIM * NVERTS * nm->E);
    nm->EToVG = asd_malloc_aligned(sizeof(uintglo_t) * NVERTS * nm->E);

    for (int r = 0; r < size; ++r)
    {
      MPI_Irecv(nm->EToVX + VDIM * NVERTS * (nm->ER + nm->recv_starts[r]),
                VDIM * NVERTS * (nm->recv_starts[r + 1] - nm->recv_starts[r]),
                MPI_DOUBLE, r, 333, comm, recv_requests + r);

      MPI_Irecv(nm->EToVG + NVERTS * (nm->ER + nm->recv_starts[r]),
                NVERTS * (nm->recv_starts[r + 1] - nm->recv_starts[r]),
                UINTGLO_MPI, r, 333, comm, recv_requests + size + r);
    }

    double *sendEToVX =
        asd_malloc_aligned(sizeof(double) * VDIM * NVERTS * nm->ES);
    uintglo_t *sendEToVG =
        asd_malloc_aligned(sizeof(uintglo_t) * NVERTS * nm->ES);

    for (uintloc_t es = 0; es < nm->ES; ++es)
    {
      const uintloc_t e = nm->ESetS[es];

      for (int n = 0; n < NVERTS; ++n)
        for (int d = 0; d < VDIM; ++d)
          sendEToVX[NVERTS * VDIM * es + VDIM * n + d] =
              om->EToVX[NVERTS * VDIM * e + VDIM * n + d];

      for (int n = 0; n < NVERTS; ++n)
        sendEToVG[NVERTS * es + n] = om->EToVG[NVERTS * e + n];
    }

    for (int r = 0; r < size; ++r)
    {
      MPI_Isend(sendEToVX + VDIM * NVERTS * nm->send_starts[r],
                VDIM * NVERTS * (nm->send_starts[r + 1] - nm->send_starts[r]),
                MPI_DOUBLE, r, 333, comm, send_requests + r);

      MPI_Isend(sendEToVG + NVERTS * nm->send_starts[r],
                NVERTS * (nm->send_starts[r + 1] - nm->send_starts[r]),
                UINTGLO_MPI, r, 333, comm, send_requests + size + r);
    }

    memcpy(nm->EToVX, om->EToVX, sizeof(double) * VDIM * NVERTS * nm->ER);
    memcpy(nm->EToVG, om->EToVG, sizeof(uintglo_t) * NVERTS * nm->ER);

    MPI_Waitall(2 * size, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(2 * size, send_requests, MPI_STATUSES_IGNORE);

    asd_free_aligned(sendEToVG);
    asd_free_aligned(sendEToVX);
  }
  // }}}

  // Check connectivity of mesh.  See if (EToE,EToF,EToO) and EToVG at least
  // agree.
  {
    int mesh_connected = 1;

    for (uintloc_t e0 = 0; e0 < nm->ER; ++e0)
    {
      for (uint8_t f0 = 0; f0 < NFACES; ++f0)
      {
        const uintloc_t e1 = nm->EToE[NFACES * e0 + f0];
        const uint8_t f1 = nm->EToF[NFACES * e0 + f0];
        const uint8_t o = nm->EToO[NFACES * e0 + f0];

        int elem_connected = 1;

        for (int v = 0; v < NFACEVERTS; ++v)
        {
          const uintglo_t vg0 =
              nm->EToVG[NVERTS * e0 +
                        FToFV[NFACEVERTS * f0 + OToFV[NFACEVERTS * o + v]]];
          const uintglo_t vg1 =
              nm->EToVG[NVERTS * e1 + FToFV[NFACEVERTS * f1 + v]];

          if (vg0 != vg1)
          {
            elem_connected = 0;
            mesh_connected = 0;
            break;
          }
        }

        if (!elem_connected)
          ASD_LERROR("Contrary to (EToE,EToF,EToO), EToVG indicates Element "
                     "%5ju Face %ju is not connected to Element %5ju Face %ju "
                     "via orientation %ju",
                     (uintmax_t)e0, (uintmax_t)f0, (uintmax_t)e1, (uintmax_t)f1,
                     (uintmax_t)o);
      }
    }

    ASD_ABORT_IF_NOT(mesh_connected, "Mesh is not connected properly.\n"
                                     "(EToE,EToF,EToO) and EToVG do not "
                                     "agree.");
  }

#if 0
  printf("ES = %ju\n", (uintmax_t)nm->ES);
  printf("ESetS\n");
  for (uintloc_t es = 0; es < nm->ES; ++es)
    printf(" %5ju\n", (uintmax_t)nm->ESetS[es]);
  printf("send_starts\n");
  for (int r = 0; r <= size; ++r)
    printf(" %5ju\n", (uintmax_t)nm->send_starts[r]);
  printf("recv_starts\n");
  for (int r = 0; r <= size; ++r)
    printf(" %5ju\n", (uintmax_t)nm->recv_starts[r]);
  printf("EE = %ju\n", (uintmax_t)nm->EE);
  printf("ESetE\n");
  for (uintloc_t ee = 0; ee < nm->EE; ++ee)
    printf(" %5ju\n", (uintmax_t)nm->ESetE[ee]);
  printf("EI = %ju\n", (uintmax_t)nm->EI);
  printf("ESetI\n");
  for (uintloc_t ei = 0; ei < nm->EI; ++ei)
    printf(" %5ju\n", (uintmax_t)nm->ESetI[ei]);
  // printf("EToV\n");
  // for (uintloc_t e = 0; e < nm->E; ++e)
  // {
  //   for (uint8_t v = 0; v < NVERTS; ++v)
  //     printf(" %4ju", (uintmax_t)nm->EToVG[NVERTS * e + v]);
  //   printf("\n");
  // }
  // printf("\n");
  printf("EToE\n");
  for (uintloc_t e = 0; e < nm->E; ++e)
  {
    for (uint8_t f = 0; f < NFACES; ++f)
      printf(" %4ju", (uintmax_t)nm->EToE[NFACES * e + f]);
    printf("\n");
  }
  printf("\n");
  printf("EToF\n");
  for (uintloc_t e = 0; e < nm->E; ++e)
  {
    for (uint8_t f = 0; f < NFACES; ++f)
      printf(" %4ju", (uintmax_t)nm->EToF[NFACES * e + f]);
    printf("\n");
  }
  printf("\n");
  printf("EToO\n");
  for (uintloc_t e = 0; e < nm->E; ++e)
  {
    for (uint8_t f = 0; f < NFACES; ++f)
      printf(" %4ju", (uintmax_t)nm->EToO[NFACES * e + f]);
    printf("\n");
  }
  printf("\n");
#endif

  asd_free(startsglo);
  asd_free(startsloc);
  asd_free_aligned(fnglo);
  asd_free_aligned(fnloc);
  asd_free_aligned(recv_requests);
  asd_free_aligned(send_requests);

  return nm;
}

static void host_mesh_write_mfem(const int rank, const char *directory,
                                 const char *prefix, const host_mesh_t *mesh)
{
  char outdir[ASD_BUFSIZ];
  struct stat sb;

  // create directory for the timestep data
  snprintf(outdir, ASD_BUFSIZ, "%s/%s", directory, prefix);
  if (stat(outdir, &sb) != 0 && mkdir(outdir, 0755) != 0 && errno != EEXIST)
    perror("making mfem directory");

  char filename[ASD_BUFSIZ];
  snprintf(filename, ASD_BUFSIZ, "%s/mesh.%06d", outdir, rank);
  ASD_VERBOSE("Writing file: '%s'", filename);

  FILE *file = fopen(filename, "w");

  if (file == NULL)
  {
    ASD_LERROR("Could not open %s for output!\n", filename);
    return;
  }

  fprintf(file, "MFEM mesh v1.0\n\n");
  fprintf(file, "dimension\n%d\n\n", VDIM);

  fprintf(file, "elements\n%" UINTLOC_PRI "\n", mesh->ER);
  for (uintloc_t e = 0; e < mesh->ER; ++e)
  {
    fprintf(file, "%" UINTLOC_PRI " %" UINTLOC_PRI, e + 1, MFEM_ELEM_TYPE);
    for (int n = 0; n < NVERTS; ++n)
      fprintf(file, " %" UINTLOC_PRI, NVERTS * e + n);

    fprintf(file, "\n");
  }
  fprintf(file, "\n");

  // Count the number of boundary faces
  uintloc_t NBC = 0;
  for (uintloc_t e = 0; e < mesh->ER; ++e)
    for (int f = 0; f < NFACES; ++f)
      if (mesh->EToE[NFACES * e + f] == e && mesh->EToF[NFACES * e + f] == f)
        ++NBC;

  fprintf(file, "boundary\n%" UINTLOC_PRI "\n", NBC);
  for (uintloc_t e = 0; e < mesh->ER; ++e)
  {
    for (int f = 0; f < NFACES; ++f)
    {
      if (mesh->EToE[NFACES * e + f] == e && mesh->EToF[NFACES * e + f] == f)
      {
        fprintf(file, "%d %" UINTLOC_PRI, 1, MFEM_FACE_TYPE);
        for (int n = 0; n < NFACEVERTS; ++n)
          fprintf(file, " %" UINTLOC_PRI,
                  NVERTS * e + FToFV[NFACEVERTS * f + n]);

        fprintf(file, "\n");
      }
    }
  }

  fprintf(file, "vertices\n%" UINTLOC_PRI "\n%d\n", NVERTS * mesh->ER, VDIM);

  for (uintloc_t e = 0; e < mesh->ER; ++e)
  {
    for (int n = 0; n < NVERTS; ++n)
    {
      fprintf(file, "        ");
      for (int d = 0; d < VDIM; ++d)
        fprintf(file, " %24.16e",
                mesh->EToVX[NVERTS * VDIM * e + VDIM * n + d]);
      fprintf(file, "\n");
    }
  }

  if (ferror(file))
  {
    ASD_LERROR("Error writing to %s\n", filename);
  }

  if (fclose(file))
  {
    ASD_LERROR("Error closing %s\n", filename);
  }
}
// }}}

// {{{ Hilbert curve partitioning
//+++++++++++++++++++++++++++ PUBLIC-DOMAIN SOFTWARE ++++++++++++++++++++++++++
// Functions: TransposetoAxes AxestoTranspose
// Purpose:   Transform in-place between Hilbert transpose and geometrical axes
// Example:   b=5 bits for each of n=3 coordinates.
//            15-bit Hilbert integer = A B C D E F G H I J K L M N O is stored
//            as its Transpose
//                   X[0] = A D G J M                X[2]|
//                   X[1] = B E H K N    <------->       | /X[1]
//                   X[2] = C F I L O               axes |/
//                          high low                     0------ X[0]
//            Axes are stored conventially as b-bit integers.
// Author: John Skilling 20 Apr 2001 to 11 Oct 2003
// doi: https://dx.doi.org/10.1063/1.1751381
//-----------------------------------------------------------------------------
#if 0
static void hilbert_trans_to_axes(uintglo_t *X)
{
  uintglo_t N = UINTGLO(2) << (UINTGLO_BITS - 1), P, Q, t;
  int i;
  // Gray decode by H ^ (H/2)
  t = X[VDIM - 1] >> 1;
  for (i = VDIM - 1; i >= 0; i--)
    X[i] ^= X[i - 1];
  X[0] ^= t;
  // Undo excess work
  for (Q = 2; Q != N; Q <<= 1)
  {
    P = Q - 1;
    for (i = VDIM - 1; i >= 0; i--)
      if (X[i] & Q)
        X[0] ^= P; // invert
      else
      {
        t = (X[0] ^ X[i]) & P;
        X[0] ^= t;
        X[i] ^= t;
      }
  } // exchange
}
#endif

static void hilbert_axes_to_trans(uintglo_t *X)
{
  uintglo_t M = UINTGLO(1) << (UINTGLO_BITS - 1), P, Q, t;
  int i;
  // Inverse undo
  for (Q = M; Q > 1; Q >>= 1)
  {
    P = Q - 1;
    for (i = 0; i < VDIM; i++)
      if (X[i] & Q)
        X[0] ^= P; // invert
      else
      {
        t = (X[0] ^ X[i]) & P;
        X[0] ^= t;
        X[i] ^= t;
      }
  } // exchange
  // Gray encode
  for (i = 1; i < VDIM; i++)
    X[i] ^= X[i - 1];
  t = 0;
  for (Q = M; Q > 1; Q >>= 1)
    if (X[VDIM - 1] & Q)
      t ^= Q - 1;
  for (i = 0; i < VDIM; i++)
    X[i] ^= t;
}

// The Transpose is stored as a Hilbert integer.  For example,
// if the 15-bit Hilbert integer = A B C D E F G H I J K L M N O is passed
// in in X as below the H gets set as below.
//
//     X[0] = A D G J M               H[0] = A B C D E
//     X[1] = B E H K N   <------->   H[1] = F G H I J
//     X[2] = C F I L O               H[2] = K L M N O
//
// TODO: replace with LUT if the code it too slow
static void hilbert_trans_to_code(const uintglo_t *X, uintglo_t *H)
{
  unsigned int i, j;
  for (i = 0; i < VDIM; ++i)
    H[i] = 0;

  for (i = 0; i < VDIM; ++i)
  {
    for (j = 0; j < UINTGLO_BITS; ++j)
    {
      const unsigned int k = i * UINTGLO_BITS + j;
      const uint64_t bit = (X[VDIM - 1 - (k % VDIM)] >> (k / VDIM)) & 1U;
      H[VDIM - 1 - i] |= (bit << j);
    }
  }
}

#define ETOH_E (VDIM)
#define ETOH_R (VDIM + 1)
#define ETOH_S (VDIM + 2)
#define NETOH (VDIM + 3)

/* qsort int comparison function */
int EToH_H_cmp(const void *a, const void *b)
{
  const uintglo_t *ia = (const uintglo_t *)a;
  const uintglo_t *ib = (const uintglo_t *)b;
  int retval = 0;

  for (int d = 0; d < VDIM; ++d)
  {
    if (ia[d] > ib[d])
      retval = 1;
    else if (ia[d] < ib[d])
      retval = -1;
    if (retval != 0)
      break;
  }

  return retval;
}

int EToH_rH_cmp(const void *a, const void *b)
{
  const uintglo_t *ia = (const uintglo_t *)a;
  const uintglo_t *ib = (const uintglo_t *)b;
  int retval = 0;

  if (ia[ETOH_R] > ib[ETOH_R])
    retval = 1;
  else if (ia[ETOH_R] < ib[ETOH_R])
    retval = -1;

  if (retval == 0)
  {
    for (int d = 0; d < VDIM; ++d)
    {
      if (ia[d] > ib[d])
        retval = 1;
      else if (ia[d] < ib[d])
        retval = -1;
      if (retval != 0)
        break;
    }
  }

  return retval;
}

// gets a partition of the mesh based on a Hilbert curve
//
// part_E - gives the number of elements per rank in the new mesh
// part_recv_starts - give the starting location into part_send_e for receiving
//                    the elements
// part_send_starts - give the starting location into part_send_e for sending
//                    the elements
// part_send_e      - give element ordering for sending the elements
static void get_hilbert_partition(MPI_Comm comm, host_mesh_t *om,
                                  uintloc_t *part_E,
                                  uintloc_t *part_recv_starts,
                                  uintloc_t *part_send_starts,
                                  uintloc_t *part_send_e)
{
  ASD_ROOT_VERBOSE("Computing Hilbert partition");

  int rank, size;
  ASD_MPI_CHECK(MPI_Comm_rank(comm, &rank));
  ASD_MPI_CHECK(MPI_Comm_size(comm, &size));

  MPI_Request *recv_requests = asd_malloc_aligned(sizeof(MPI_Request) * size);
  MPI_Request *send_requests = asd_malloc_aligned(sizeof(MPI_Request) * size);

  // {{{ compute centroid of each element
  double *EToC = asd_malloc_aligned(sizeof(double) * VDIM * om->E);
  double cmaxloc[VDIM] = {-DBL_MAX}, cminloc[VDIM] = {DBL_MAX};
  for (uintloc_t e = 0; e < om->E; ++e)
  {
    for (int d = 0; d < VDIM; ++d)
      EToC[e * VDIM + d] = 0;

    for (int n = 0; n < NVERTS; ++n)
      for (int d = 0; d < VDIM; ++d)
        EToC[e * VDIM + d] += om->EToVX[NVERTS * VDIM * e + VDIM * n + d];

    for (int d = 0; d < VDIM; ++d)
      EToC[e * VDIM + d] /= NVERTS;

    for (int d = 0; d < VDIM; ++d)
    {
      cmaxloc[d] = ASD_MAX(cmaxloc[d], EToC[e * VDIM + d]);
      cminloc[d] = ASD_MIN(cminloc[d], EToC[e * VDIM + d]);
    }
  }
// }}}

#if 0
  printf("EToC:\n");
  for (uintloc_t e = 0; e < om->E; ++e)
  {
    printf("%20" UINTLOC_PRI, e);
    for (int d = 0; d < VDIM; ++d)
      printf(" %24.16e", EToC[e * VDIM + d]);
    printf("\n");
  }
#endif

  // {{{ compute Hilbert integer centroids
  double cmaxglo[VDIM] = {-DBL_MAX}, cminglo[VDIM] = {DBL_MAX};
  // These calls could be joined
  ASD_MPI_CHECK(
      MPI_Allreduce(cmaxloc, cmaxglo, VDIM, MPI_DOUBLE, MPI_MAX, comm));
  ASD_MPI_CHECK(
      MPI_Allreduce(cminloc, cminglo, VDIM, MPI_DOUBLE, MPI_MIN, comm));

#if 0
  printf("min:\n");
  for (int d = 0; d < VDIM; ++d)
    printf(" %24.16e", cminglo[d]);
  printf("\n");

  printf("max:\n");
  for (int d = 0; d < VDIM; ++d)
    printf(" %24.16e", cmaxglo[d]);
  printf("\n");
#endif

  // convert the centroids to hilbert integer
  uintglo_t *EToHloc = asd_malloc_aligned(sizeof(uintglo_t) * NETOH * om->E);

#if 0
  printf("uintglo_t max: %20" UINTGLO_PRI "\n", UINTGLO_MAX);
  printf("EToCI:\n");
#endif

  for (uintloc_t e = 0; e < om->E; ++e)
  {
    uintglo_t X[VDIM], H[VDIM];

    // TODO Is there a better way to get integer coordinates?
    for (int d = 0; d < VDIM; ++d)
    {
      long double x =
          (EToC[e * VDIM + d] - cminglo[d]) / (cmaxglo[d] - cminglo[d]);
      X[d] = (uintglo_t)(x * UINTGLO_MAX);
    }

#if 0
    printf("%20" UINTLOC_PRI, e);
    for (int d = 0; d < VDIM; ++d)
      printf(" %20" UINTGLO_PRI, X[d]);
    printf("\n");
#endif

    hilbert_axes_to_trans(X);
    hilbert_trans_to_code(X, H);

    for (int d = 0; d < VDIM; ++d)
      EToHloc[e * NETOH + d] = H[d];
    EToHloc[e * NETOH + ETOH_E] = e;           // original element number
    EToHloc[e * NETOH + ETOH_R] = rank;        // original rank
    EToHloc[e * NETOH + ETOH_S] = UINTGLO_MAX; // new rank
  }
  asd_free_aligned(EToC);

#if 0
  printf("EToHloc before sort:\n");
  for (uintloc_t e = 0; e < om->E; ++e)
  {
    for (int n = 0; n < NETOH; ++n)
      printf(" %20" UINTGLO_PRI, EToHloc[e * NETOH + n]);
    printf("\n");
  }
#endif
  // }}}

  qsort(EToHloc, om->E, sizeof(uintglo_t) * NETOH, EToH_H_cmp);

#if 0
  printf("EToHloc:\n");
  for (uintloc_t e = 0; e < om->E; ++e)
  {
    for (int n = 0; n < NETOH; ++n)
      printf(" %20" UINTGLO_PRI, EToHloc[e * NETOH + n]);
    printf("\n");
  }
#endif

  // {{{ select pivots
  uintglo_t *pivotsloc, *pivotsglo;

  pivotsloc = asd_malloc_aligned(sizeof(uintglo_t) * NETOH * size);
  pivotsglo = (rank == 0)
                  ? asd_malloc_aligned(sizeof(uintglo_t) * NETOH * size * size)
                  : NULL;

  for (int r = 0; r < size; ++r)
    for (int d = 0; d < VDIM; ++d)
      pivotsloc[r * NETOH + d] = EToHloc[((om->E * r) / size) * NETOH + d];

#if 0
  printf("pivotsloc\n");
  for (int r = 0; r < size; ++r)
  {
    printf("%2d ", r);
    for (int d = 0; d < VDIM; ++d)
      printf(" %20" UINTGLO_PRI, pivotsloc[r * NETOH + d]);
    printf("\n");
  }
#endif

  ASD_MPI_CHECK(MPI_Gather(pivotsloc, size * NETOH, UINTGLO_MPI, pivotsglo,
                           size * NETOH, UINTGLO_MPI, 0, comm));

#if 0
  if (rank == 0)
  {
    printf("pivotsglo\n");
    for (int r = 0; r < size; ++r)
      for (int s = 0; s < size; ++s)
      {
        printf("%2d %2d ", r, s);
        for (int d = 0; d < VDIM; ++d)
          printf(" %20" UINTGLO_PRI,
                 pivotsglo[r * size * NETOH + s * NETOH + d]);
        printf("\n");
      }
  }
#endif

  if (rank == 0)
  {
    uintglo_t *sorted_pivotsglo =
        asd_malloc_aligned(sizeof(uintglo_t) * NETOH * size * size);
    asd_multimergesort(sorted_pivotsglo, pivotsglo, size, size,
                       sizeof(uintglo_t) * NETOH, EToH_H_cmp);

#if 0
    printf("sorted_pivotsglo\n");
    for (int r = 0; r < size; ++r)
      for (int s = 0; s < size; ++s)
      {
        printf("%2d %2d ", r, s);
        for (int d = 0; d < VDIM; ++d)
          printf(" %20" UINTGLO_PRI,
                 sorted_pivotsglo[r * size * NETOH + s * NETOH + d]);
        printf("\n");
      }
#endif

    for (int r = 0; r < size; ++r)
      for (int d = 0; d < VDIM; ++d)
        pivotsloc[r * NETOH + d] =
            sorted_pivotsglo[((size * size * r) / size) * NETOH + d];

    asd_free_aligned(pivotsglo);
    asd_free_aligned(sorted_pivotsglo);
  }

  ASD_MPI_CHECK(MPI_Bcast(pivotsloc, size * NETOH, UINTGLO_MPI, 0, comm));
  // }}}

  // {{{ compute communication map to globally sort EToH
  uintloc_t *startsloc = asd_calloc(sizeof(uintloc_t), size + 1);
  startsloc[size] = om->E;

  // binary search for the starts of each rank in EToHloc
  for (int r = 0; r < size; ++r)
  {
    uintloc_t start = 0;
    uintloc_t end = om->E - 1;
    uintloc_t offset = 0;

    if (om->E > 0)
    {
      while (end >= start)
      {
        offset = (start + end) / 2;

        if (offset == 0)
          break;

        int c = EToH_H_cmp(EToHloc + offset * NETOH, pivotsloc + r * NETOH);

        if (start == end)
        {
          if (c < 0)
            ++offset;
          break;
        }

        if (c < 0)
          start = offset + 1;
        else if (c > 0)
          end = offset - 1;
        else
          break;
      }
    }

    ASD_ABORT_IF_NOT(offset <= om->E, "Problem with binary search");

    startsloc[r] = offset;
  }
  asd_free_aligned(pivotsloc);

#if 0
  printf("startsloc\n");
  for (int r = 0; r <= size; ++r)
  {
    printf("%2d %20" UINTLOC_PRI, r, startsloc[r]);
    if (startsloc[r] < om->E)
      for (int d = 0; d < VDIM; ++d)
        printf(" %20" UINTGLO_PRI, EToHloc[startsloc[r] * NETOH + d]);
    printf("\n");
  }
  printf("\n\n");
#endif

  // get number of elements to receive
  int *countsglo = asd_malloc(sizeof(int) * size);
  for (int r = 0; r < size; ++r)
  {
    uintmax_t uic = startsloc[r + 1] - startsloc[r];
    ASD_ABORT_IF_NOT(uic < INT_MAX, "Sending more than INT_MAX elements");
    int c = (int)uic;
    ASD_MPI_CHECK(MPI_Gather(&c, 1, MPI_INT, countsglo, 1, MPI_INT, r, comm));
  }
  uintloc_t *startsglo = asd_calloc(sizeof(uintloc_t), size + 1);
  for (int r = 0; r < size; ++r)
    startsglo[r + 1] = startsglo[r] + countsglo[r];
  asd_free(countsglo);

#if 0
  printf("countsglo\n");
  for (int r = 0; r < size; ++r)
    printf("%2d %20ju\n", r, (uintmax_t)(startsglo[r + 1] - startsglo[r]));

  printf("countsloc\n");
  for (int r = 0; r < size; ++r)
    printf("%2d %20ju\n", r, (uintmax_t)(startsloc[r + 1] - startsloc[r]));
#endif
  // }}}

  // {{{ receive EToH for global sort
  uintglo_t *EToHglo =
      asd_malloc_aligned(sizeof(uintglo_t) * NETOH * startsglo[size]);

  for (int r = 0; r < size; ++r)
    MPI_Irecv(EToHglo + NETOH * startsglo[r],
              NETOH * (startsglo[r + 1] - startsglo[r]), UINTGLO_MPI, r, 333,
              comm, recv_requests + r);

  for (int r = 0; r < size; ++r)
    MPI_Isend(EToHloc + NETOH * startsloc[r],
              NETOH * (startsloc[r + 1] - startsloc[r]), UINTGLO_MPI, r, 333,
              comm, send_requests + r);

  MPI_Waitall(size, recv_requests, MPI_STATUSES_IGNORE);
  MPI_Waitall(size, send_requests, MPI_STATUSES_IGNORE);

#if 0
  printf("EToHglo received:\n");
  for (uintloc_t e = 0; e < startsglo[size]; ++e)
  {
    for (int n = 0; n < NETOH; ++n)
      printf(" %20" UINTGLO_PRI, EToHglo[e * NETOH + n]);
    printf("\n");
  }
#endif
  // }}}

  // TODO Should we replace with multi-mergesort?
  qsort(EToHglo, startsglo[size], sizeof(uintglo_t) * NETOH, EToH_H_cmp);

  // {{{ compute destination rank for each element
  uintglo_t *Eglo_first_index = asd_malloc(sizeof(uintglo_t) * (size + 1));
  {
    uintglo_t Eglo = startsglo[size];
    uintglo_t *Eglo_in_proc = asd_malloc(sizeof(uintglo_t) * 2 * size);

    ASD_MPI_CHECK(MPI_Allgather(&Eglo, 1, UINTGLO_MPI, Eglo_in_proc, 1,
                                UINTGLO_MPI, comm));

    ASD_ABORT_IF_NOT(size > 0, "We need at least one MPI rank");
    Eglo_first_index[0] = 0;
    for (int r = 0; r < size; ++r)
      Eglo_first_index[r + 1] = Eglo_in_proc[r] + Eglo_first_index[r];

    asd_free(Eglo_in_proc);
  }

  const uintglo_t ms = Eglo_first_index[rank];
  const uintglo_t me = Eglo_first_index[rank + 1];

  for (int r = 0; r < size; ++r)
  {
    const uintglo_t Etotal = Eglo_first_index[size];
    const uintglo_t ns = linpart_starting_row(r, size, Etotal);
    const uintglo_t ne = linpart_starting_row(r + 1, size, Etotal);

    if (ns < me && ne > ms)
    {
      const uintglo_t start = ASD_MAX(ns, ms);
      const uintglo_t end = ASD_MIN(ne, me);
      for (uintglo_t e = start; e < end; ++e)
        EToHglo[(e - ms) * NETOH + ETOH_S] = r;
    }
  }

// }}}

#if 0
  printf("EToHglo sorted:\n");
  for (uintloc_t e = 0; e < startsglo[size]; ++e)
  {
    for (int n = 0; n < NETOH; ++n)
      printf(" %20" UINTGLO_PRI, EToHglo[e * NETOH + n]);
    printf("\n");
  }
#endif

  qsort(EToHglo, startsglo[size], sizeof(uintglo_t) * NETOH, EToH_rH_cmp);

#if 0
  printf("EToHglo sorted to send:\n");
  for (uintloc_t e = 0; e < startsglo[size]; ++e)
  {
    for (int n = 0; n < NETOH; ++n)
      printf(" %20" UINTGLO_PRI, EToHglo[e * NETOH + n]);
    printf("\n");
  }
#endif

  // {{{ receive EToH with partition information
  for (int r = 0; r < size; ++r)
    MPI_Irecv(EToHloc + NETOH * startsloc[r],
              NETOH * (startsloc[r + 1] - startsloc[r]), UINTGLO_MPI, r, 333,
              comm, recv_requests + r);

  for (int r = 0; r < size; ++r)
    MPI_Isend(EToHglo + NETOH * startsglo[r],
              NETOH * (startsglo[r + 1] - startsglo[r]), UINTGLO_MPI, r, 333,
              comm, send_requests + r);

  MPI_Waitall(size, recv_requests, MPI_STATUSES_IGNORE);
  MPI_Waitall(size, send_requests, MPI_STATUSES_IGNORE);

#if 0
  printf("EToHloc:\n");
  for (uintloc_t e = 0; e < om->E; ++e)
  {
    for (int n = 0; n < NETOH; ++n)
      printf(" %20" UINTGLO_PRI, EToHloc[e * NETOH + n]);
    printf("\n");
  }
#endif
  // }}}

  uintloc_t *part_send_counts = asd_calloc(sizeof(uintloc_t), size);
  for (uintloc_t e = 0; e < om->E; ++e)
  {
    ASD_ASSERT(EToHloc[e * NETOH + ETOH_E] < UINTLOC_MAX);

    ASD_ABORT_IF_NOT(EToHloc[e * NETOH + ETOH_S] < UINTLOC_MAX,
                     "Lost element %ju in Hilbert partition",
                     EToHloc[e * NETOH + ETOH_E]);

    part_send_e[e] = (uintloc_t)EToHloc[e * NETOH + ETOH_E];
    ++part_send_counts[EToHloc[e * NETOH + ETOH_S]];
  }

  part_send_starts[0] = 0;
  for (int r = 0; r < size; ++r)
    part_send_starts[r + 1] = part_send_starts[r] + part_send_counts[r];

  uintloc_t *part_recv_counts = asd_calloc(sizeof(uintloc_t), size);
  for (int r = 0; r < size; ++r)
  {
    uintmax_t uic = part_send_starts[r + 1] - part_send_starts[r];
    ASD_ABORT_IF_NOT(uic < UINTLOC_MAX,
                     "Sending more than UINTLOC_MAX elements");
    uintloc_t c = (uintloc_t)uic;
    ASD_MPI_CHECK(MPI_Gather(&c, 1, UINTLOC_MPI, part_recv_counts, 1,
                             UINTLOC_MPI, r, comm));
  }

  part_recv_starts[0] = 0;
  for (int r = 0; r < size; ++r)
    part_recv_starts[r + 1] = part_recv_starts[r] + part_recv_counts[r];

  for (int r = 0; r < size; ++r)
  {
    const uintmax_t Etotal = Eglo_first_index[size];
    const uintmax_t Eloc = linpart_local_num_rows(r, size, Etotal);

    ASD_ABORT_IF_NOT(Eloc < UINTLOC_MAX, "Too many local elements %ju", Eloc);
    part_E[r] = (uintloc_t)Eloc;
  }

  asd_free(Eglo_first_index);
  asd_free(part_send_counts);
  asd_free(part_recv_counts);
  asd_free(startsglo);
  asd_free(startsloc);
  asd_free_aligned(EToHglo);
  asd_free_aligned(EToHloc);
  asd_free_aligned(recv_requests);
  asd_free_aligned(send_requests);
}

static host_mesh_t *partition(MPI_Comm comm, const host_mesh_t *om,
                              const uintloc_t *part_E,
                              const uintloc_t *part_recv_starts,
                              const uintloc_t *part_send_starts,
                              const uintloc_t *part_send_e)
{
  int rank, size;
  ASD_MPI_CHECK(MPI_Comm_rank(comm, &rank));
  ASD_MPI_CHECK(MPI_Comm_size(comm, &size));

  MPI_Request *recv_requests =
      asd_malloc_aligned(sizeof(MPI_Request) * 2 * size);
  MPI_Request *send_requests =
      asd_malloc_aligned(sizeof(MPI_Request) * 2 * size);

  host_mesh_t *nm = asd_malloc(sizeof(host_mesh_t));

  uintglo_t *bufEToVG = asd_malloc_aligned(sizeof(uintglo_t) * NVERTS * om->E);
  double *bufEToVX = asd_malloc_aligned(sizeof(double) * NVERTS * VDIM * om->E);

  nm->E = part_E[rank];
  nm->EToVG = asd_malloc_aligned(sizeof(uintglo_t) * NVERTS * nm->E);
  nm->EToVX = asd_malloc_aligned(sizeof(double) * NVERTS * VDIM * nm->E);

  ASD_VERBOSE("Partition mesh and keep %ju elements.", (intmax_t)nm->E);

#if 0
  printf("part_E:\n");
  for (int r = 0; r < size; ++r)
    printf("%20" UINTLOC_PRI " %20" UINTLOC_PRI "\n", r, part_E[r]);

  printf("part_recv_starts:\n");
  for (int r = 0; r <= size; ++r)
    printf("%20" UINTLOC_PRI " %20" UINTLOC_PRI "\n", r, part_recv_starts[r]);

  printf("part_send_starts:\n");
  for (int r = 0; r <= size; ++r)
    printf("%20" UINTLOC_PRI " %20" UINTLOC_PRI "\n", r, part_send_starts[r]);

  printf("part_send_e:\n");
  for (uintloc_t e = 0; e < om->E; ++e)
    printf("%20" UINTLOC_PRI " %20" UINTLOC_PRI "\n", e, part_send_e[e]);

  for (int r = 0; r < size; ++r)
    printf("%2d<-%2d : %20" UINTLOC_PRI "\n", rank, r,
           part_recv_starts[r + 1] - part_recv_starts[r]);

  for (int r = 0; r < size; ++r)
    printf("%2d->%2d : %20" UINTLOC_PRI "\n", rank, r,
           part_send_starts[r + 1] - part_send_starts[r]);

#endif

  for (uintloc_t e1 = 0; e1 < om->E; ++e1)
  {
    const uintloc_t e0 = part_send_e[e1];

    for (int n = 0; n < NVERTS; ++n)
      bufEToVG[NVERTS * e1 + n] = om->EToVG[NVERTS * e0 + n];
  }

  for (uintloc_t e1 = 0; e1 < om->E; ++e1)
  {
    const uintloc_t e0 = part_send_e[e1];

    for (int n = 0; n < NVERTS; ++n)
      for (int d = 0; d < VDIM; ++d)
        bufEToVX[NVERTS * VDIM * e1 + VDIM * n + d] =
            om->EToVX[NVERTS * VDIM * e0 + VDIM * n + d];
  }

  for (int r = 0; r < size; ++r)
    MPI_Irecv(nm->EToVG + NVERTS * part_recv_starts[r],
              NVERTS * (part_recv_starts[r + 1] - part_recv_starts[r]),
              UINTGLO_MPI, r, 333, comm, recv_requests + r);

  for (int r = 0; r < size; ++r)
    MPI_Irecv(nm->EToVX + NVERTS * VDIM * part_recv_starts[r],
              NVERTS * VDIM * (part_recv_starts[r + 1] - part_recv_starts[r]),
              MPI_DOUBLE, r, 333, comm, recv_requests + size + r);

  for (int r = 0; r < size; ++r)
    MPI_Isend(bufEToVG + NVERTS * part_send_starts[r],
              NVERTS * (part_send_starts[r + 1] - part_send_starts[r]),
              UINTGLO_MPI, r, 333, comm, send_requests + r);

  for (int r = 0; r < size; ++r)
    MPI_Isend(bufEToVX + NVERTS * VDIM * part_send_starts[r],
              NVERTS * VDIM * (part_send_starts[r + 1] - part_send_starts[r]),
              MPI_DOUBLE, r, 333, comm, send_requests + size + r);

  MPI_Waitall(2 * size, recv_requests, MPI_STATUSES_IGNORE);
  MPI_Waitall(2 * size, send_requests, MPI_STATUSES_IGNORE);

  asd_free_aligned(bufEToVG);
  asd_free_aligned(bufEToVX);
  asd_free_aligned(recv_requests);
  asd_free_aligned(send_requests);

  // Initialize an unconnected mesh
  nm->ER = nm->E;
  nm->EG = 0;

  nm->EToE = asd_malloc_aligned(sizeof(uintloc_t) * NFACES * nm->E);
  nm->EToF = asd_malloc_aligned(sizeof(uint8_t) * NFACES * nm->E);
  nm->EToO = asd_malloc_aligned(sizeof(uint8_t) * NFACES * nm->E);

  for (uintloc_t e = 0; e < nm->E; ++e)
  {
    for (uint8_t f = 0; f < NFACES; ++f)
    {
      nm->EToE[NFACES * e + f] = e;
      nm->EToF[NFACES * e + f] = f;
      nm->EToO[NFACES * e + f] = 0;
    }
  }

  nm->EI = 0;
  nm->EE = 0;
  nm->ES = 0;

  nm->ESetI = asd_malloc_aligned(sizeof(uintloc_t) * nm->EI);
  nm->ESetE = asd_malloc_aligned(sizeof(uintloc_t) * nm->EE);
  nm->ESetS = asd_malloc_aligned(sizeof(uintloc_t) * nm->ES);

  nm->recv_starts = asd_malloc_aligned(sizeof(uintloc_t) * (size + 1));
  nm->send_starts = asd_malloc_aligned(sizeof(uintloc_t) * (size + 1));

  for (int r = 0; r <= size; ++r)
  {
    nm->recv_starts[r] = 0;
    nm->send_starts[r] = 0;
  }
  nm->recv_starts[rank + 1] = nm->E;

  return nm;
}
// }}}

// {{{ App
typedef struct app
{

  double rk4a[5];
  double rk4b[5];
  double rk4c[6];

  prefs_t *prefs;
  occaDevice device;
  occaStream copy;
  occaStream cmdx;

  host_mesh_t *hm;
  host_operators_t *hops;

  occaMemory vgeo;
  occaMemory vfgeo;
  occaMemory fgeo;
  occaMemory Jq;
  occaMemory wq;

  occaMemory mapPq;
  occaMemory Fmask;

  occaMemory nrJ, nsJ, ntJ;
  occaMemory Drq, Dsq, Dtq;
  occaMemory Drstq;

  occaMemory Vq;
  occaMemory Pq;

  occaMemory VqLq;
  occaMemory VqPq;
  occaMemory VfPq;
  occaMemory Vfqf;

  occaMemory Q, Qf, rhsQf, rhsQ, resQ;

  //Testing
#ifdef TESTING
  occaMemory rhoLog;
  occaMemory betaLog;
  occaMemory storage;
  occaMemory store_pNq;
  occaMemory store_pNfqNfaces;
#endif

  occaKernelInfo info;

  occaKernel vol;
  occaKernel surf;
  occaKernel update;
  occaKernel test;
  occaKernel aux;
#ifdef TESTING
  occaKernel logmean;
  occaKernel vol_sub1;
  occaKernel vol_sub2;
#endif
} app_t;

// containers for U,V and coordinates
typedef struct
{
  dfloat_t x;
  dfloat_t y;
  dfloat_t z;
} coord;
typedef struct
{
  dfloat_t U1;
  dfloat_t U2;
  dfloat_t U3;
  dfloat_t U4;
  dfloat_t U5;

  dfloat_t rho;
  dfloat_t rhou;
  dfloat_t rhov;
  dfloat_t rhow;
  dfloat_t E;
} euler_fields;

static app_t *app_new(const char *prefs_filename, MPI_Comm comm)
{
  app_t *app = asd_malloc(sizeof(app_t));

  //
  // Preferences
  //
  app->prefs = prefs_new(prefs_filename, comm);
  prefs_print(app->prefs);

  //
  // OCCA
  //
  app->device = occaCreateDevice(app->prefs->occa_info);
  if (app->prefs->occa_flags)
    occaDeviceSetCompilerFlags(app->device, app->prefs->occa_flags);

  app->copy = occaDeviceCreateStream(app->device);
  app->cmdx = occaDeviceCreateStream(app->device);
  occaDeviceSetStream(app->device, app->cmdx);

  //
  // Read and partition mesh
  //
  host_mesh_t *m, *n;
  m = host_mesh_read_msh(app->prefs);
  printf("mesh reading done\n");

  host_mesh_write_mfem(app->prefs->rank, app->prefs->output_datadir, "mesh_pre",
                       m);

  if (app->prefs->mesh_sfc_partition)
  {
    uintloc_t *part_E =
        asd_malloc_aligned(sizeof(uintloc_t) * (app->prefs->size + 1));
    uintloc_t *part_recv_starts =
        asd_malloc_aligned(sizeof(uintloc_t) * (app->prefs->size + 1));
    uintloc_t *part_send_starts =
        asd_malloc_aligned(sizeof(uintloc_t) * (app->prefs->size + 1));
    uintloc_t *part_send_e = asd_malloc_aligned(sizeof(uintloc_t) * m->E);

    get_hilbert_partition(app->prefs->comm, m, part_E, part_recv_starts,
                          part_send_starts, part_send_e);
    n = partition(app->prefs->comm, m, part_E, part_recv_starts,
                  part_send_starts, part_send_e);

    asd_free_aligned(part_E);
    asd_free_aligned(part_recv_starts);
    asd_free_aligned(part_send_starts);
    asd_free_aligned(part_send_e);
    host_mesh_free(m);
    asd_free(m);

    m = n;
  }

  n = host_mesh_connect(app->prefs->comm, m);

  host_mesh_free(m);
  asd_free(m);

  app->hm = n;

  host_mesh_write_mfem(app->prefs->rank, app->prefs->output_datadir, "mesh",
                       app->hm);

// foo(0);
#if ELEM_TYPE == 0 // triangle
  app->hops = host_operators_new_2D(app->prefs->mesh_N, app->prefs->mesh_M,
                                    app->hm->E, app->hm->EToE, app->hm->EToF,
                                    app->hm->EToO, app->hm->EToVX);
#else
  app->hops = host_operators_new_3D(app->prefs->mesh_N, app->prefs->mesh_M,
                                    app->hm->E, app->hm->EToE, app->hm->EToF,
                                    app->hm->EToO, app->hm->EToVX);
#endif

  // Allocate/fill data on the device
  const uintloc_t E = app->hm->E;
  const int Np = app->hops->Np;
  const int Nq = app->hops->Nq;
  printf("%d elements in mesh, degree %d, degree %d cubature.\n",E,app->prefs->mesh_N,app->prefs->mesh_M);
  const int Nfp = app->hops->Nfp;
  const int Nfq = app->hops->Nfq;
  const int Nfaces = app->hops->Nfaces;
  const int Nvgeo = app->hops->Nvgeo;
  const int Nfgeo = app->hops->Nfgeo;

  app->rk4a[0] = 0.0;
  app->rk4a[1] = -567301805773.0 / 1357537059087.0;
  app->rk4a[2] = -2404267990393.0 / 2016746695238.0;
  app->rk4a[3] = -3550918686646.0 / 2091501179385.0;
  app->rk4a[4] = -1275806237668.0 / 842570457699.0;

  app->rk4b[0] = 1432997174477.0 / 9575080441755.0;
  app->rk4b[1] = 5161836677717.0 / 13612068292357.0;
  app->rk4b[2] = 1720146321549.0 / 2090206949498.0;
  app->rk4b[3] = 3134564353537.0 / 4481467310338.0;
  app->rk4b[4] = 2277821191437.0 / 14882151754819.0;

  app->rk4c[0] = 0.0;
  app->rk4c[1] = 1432997174477.0 / 9575080441755.0;
  app->rk4c[2] = 2526269341429.0 / 6820363962896.0;
  app->rk4c[3] = 2006345519317.0 / 3224310063776.0;
  app->rk4c[4] = 2802321613138.0 / 2924317926251.0;
  app->rk4c[5] = 1.0;

  app->vgeo = device_malloc(app->device, sizeof(dfloat_t) * Nq * Nvgeo * E,
                            app->hops->vgeo);

  app->vfgeo = device_malloc(app->device, sizeof(dfloat_t) * Nfq * Nfaces * Nvgeo * E,
			     app->hops->vfgeo);

  app->fgeo =
      device_malloc(app->device, sizeof(dfloat_t) * Nfq * Nfaces * Nfgeo * E,
                    app->hops->fgeo);
  app->Jq =
      device_malloc(app->device, sizeof(dfloat_t) * Nq * E, app->hops->Jq);

  app->wq =
      device_malloc(app->device, sizeof(dfloat_t) * Nq, app->hops->wq);

  app->mapPq = device_malloc(app->device, sizeof(uintloc_t) * Nfq * Nfaces * E,
                             app->hops->mapPq);

  app->Fmask = device_malloc(app->device, sizeof(uintloc_t) * Nfp * Nfaces,
                             app->hops->Fmask);

  app->nrJ = device_malloc(app->device, sizeof(dfloat_t) * Nfq * Nfaces,
                           app->hops->nrJ);
  app->nsJ = device_malloc(app->device, sizeof(dfloat_t) * Nfq * Nfaces,
                           app->hops->nsJ);

#if VDIM == 3
  app->ntJ = device_malloc(app->device, sizeof(dfloat_t) * Nfq * Nfaces,
                           app->hops->ntJ);
#endif

  app->Drq =
      device_malloc(app->device, sizeof(dfloat_t) * Nq * Nq, app->hops->Drq);
  app->Dsq =
      device_malloc(app->device, sizeof(dfloat_t) * Nq * Nq, app->hops->Dsq);

#if VDIM == 3
  app->Dtq =
      device_malloc(app->device, sizeof(dfloat_t) * Nq * Nq, app->hops->Dtq);

  app->Drstq = device_malloc(app->device, sizeof(dfloat_t) * 3 * Nq * Nq,
                             app->hops->Drstq);
#endif

  app->Vq =
      device_malloc(app->device, sizeof(dfloat_t) * Np * Nq, app->hops->Vq);
  app->Pq =
      device_malloc(app->device, sizeof(dfloat_t) * Np * Nq, app->hops->Pq);

  app->VqLq = device_malloc(app->device, sizeof(dfloat_t) * Nfq * Nfaces * Nq,
                            app->hops->VqLq);
  app->VqPq =
      device_malloc(app->device, sizeof(dfloat_t) * Nq * Nq, app->hops->VqPq);
  app->VfPq = device_malloc(app->device, sizeof(dfloat_t) * Nfq * Nfaces * Nq,
                            app->hops->VfPq);
  app->Vfqf =
      device_malloc(app->device, sizeof(dfloat_t) * Nfq * Nfaces * Nfp * Nfaces,
                    app->hops->Vfqf);

  app->Q =
      device_malloc(app->device, sizeof(dfloat_t) * Nq * NFIELDS * E, NULL);
  app->rhsQ =
      device_malloc(app->device, sizeof(dfloat_t) * Nq * NFIELDS * E, NULL);
  app->resQ =
      device_malloc(app->device, sizeof(dfloat_t) * Nq * NFIELDS * E, NULL);

  // rhsQf = temp storage
  app->rhsQf = device_malloc(
      app->device, sizeof(dfloat_t) * Nfq * Nfaces * NFIELDS * E, NULL);
  app->Qf = device_malloc(app->device,
                          sizeof(dfloat_t) * Nfq * Nfaces * NFIELDS * E, NULL);

  // Testing
#ifdef TESTING
  app->rhoLog = device_malloc(app->device, sizeof(dfloat_t) * Nq * Nfq * Nfaces * app->prefs->kernel_KblkV * E, NULL);
  app->betaLog = device_malloc(app->device, sizeof(dfloat_t) * Nq * Nfq * Nfaces * app->prefs->kernel_KblkV * E, NULL);
  app->storage = device_malloc(app->device, sizeof(dfloat_t) * Nq * Nfq * Nfaces * NFIELDS * app->prefs->kernel_KblkV * E, NULL);
  app->store_pNq = device_malloc(app->device, sizeof(dfloat_t) * NFIELDS * Nq * Nfq * Nfaces * app->prefs->kernel_KblkV * E, NULL);
  app->store_pNfqNfaces = device_malloc(app->device, sizeof(dfloat_t) * NFIELDS * Nq * Nfq * Nfaces * app->prefs->kernel_KblkV * E, NULL);
#endif

  // Fill info for kernels
  occaKernelInfo info = occaCreateKernelInfo();
  occaKernelInfoAddDefine(info, "uintloc_t", occaString(occa_uintloc_name));
  const char *const dfloat =
      (sizeof(double) == sizeof(dfloat_t)) ? "double" : "float";
  occaKernelInfoAddDefine(info, "dfloat", occaString(dfloat));
  if (sizeof(double) == sizeof(dfloat_t))
    occaKernelInfoAddDefine(info, "p_DFLOAT_DOUBLE", occaInt(1));
  else
    occaKernelInfoAddDefine(info, "p_DFLOAT_FLOAT", occaInt(1));

  occaKernelInfoAddDefine(info, "p_DFLOAT_MAX", occaDfloat(DFLOAT_MAX));

  occaKernelInfoAddDefine(info, "p_KblkU", occaUInt(app->prefs->kernel_KblkU));
  occaKernelInfoAddDefine(info, "p_KblkV", occaUInt(app->prefs->kernel_KblkV));
  occaKernelInfoAddDefine(info, "p_KblkS", occaUInt(app->prefs->kernel_KblkS));
  occaKernelInfoAddDefine(info, "p_KblkF", occaUInt(app->prefs->kernel_KblkF));
  occaKernelInfoAddDefine(info, "p_TAU", occaDfloat(app->prefs->tau));

  const int T = ASD_MAX(Nq, Nfq * Nfaces);
  occaKernelInfoAddDefine(info, "p_T", occaUInt(T));

  occaKernelInfoAddDefine(info, "p_gamma",
                          occaDfloat(app->prefs->physical_gamma));

  occaKernelInfoAddDefine(info, "p_Np", occaUInt(Np));

  occaKernelInfoAddDefine(info, "p_Nq", occaUInt(Nq));
  int log2Nq = (int)ceil(log2(Nq));
  int ceilNq2 = (int) pow(2,log2Nq);
  //  printf("Nq = %d, ceilNq2 = %d\n",Nq,ceilNq2);
  occaKernelInfoAddDefine(info, "p_log2Nq", occaUInt(log2Nq));
  occaKernelInfoAddDefine(info, "p_ceilNq2", occaUInt(ceilNq2));  // closest power of 2
  occaKernelInfoAddDefine(info, "p_NfqNfaces", occaUInt(Nfq * Nfaces));
  occaKernelInfoAddDefine(info, "p_NfpNfaces", occaUInt(Nfp * Nfaces));
  occaKernelInfoAddDefine(info, "p_Nvgeo", occaUInt(Nvgeo));
  occaKernelInfoAddDefine(info, "p_Nfgeo", occaUInt(Nfgeo));
  occaKernelInfoAddDefine(info, "p_Nfaces", occaUInt(Nfaces));
  occaKernelInfoAddDefine(info, "p_Nfields", occaUInt(NFIELDS));
  occaKernelInfoAddDefine(info, "p_Nfq", occaUInt(Nfq));
  occaKernelInfoAddDefine(info, "p_Nfp", occaUInt(Nfp));

//==================================================================
// Testing
//==================================================================
  const int pT = (1 + ((T-1) >> 5)) * 32;
  const int pNfaces = (1 + ((Nfaces-1) >> 5)) * 32;
  const int pNq = (1 + ((Nq-1) >> 5)) * 32;
  const int pNfqNfaces = (1 + (((Nfq * Nfaces)-1) >> 5)) * 32;
  const int pSum = pT + pNfaces + pNq + pNfqNfaces;
  occaKernelInfoAddDefine(info, "pT32", occaUInt(pT));
  occaKernelInfoAddDefine(info, "pNfaces32", occaUInt(pNfaces));
  occaKernelInfoAddDefine(info, "pNq32", occaUInt(pNq));
  occaKernelInfoAddDefine(info, "pNfqNfaces32", occaUInt(pNfqNfaces));
  occaKernelInfoAddDefine(info, "pSum32", occaUInt(pSum));
  occaKernelInfoAddDefine(info, "E_test", occaUInt(E));
//==================================================================

  // Add rank to the kernels as a workaround for occa cache issues of
  // having multiple processes trying to use the same kernel source.
  occaKernelInfoAddDefine(info, "p_RANK", occaInt(app->prefs->rank));

  if (sizeof(dfloat_t) == 4)
  {
    occaKernelInfoAddDefine(info, "USE_DOUBLE", occaInt(0));
  }
  else
  {
    occaKernelInfoAddDefine(info, "USE_DOUBLE", occaInt(1));
  }

  app->info = info;

// TODO build kernels

#if VDIM == 2 // triangle
  printf("building 2D kernels\n");
  //  app->vol = occaDeviceBuildKernelFromSource(app->device, "okl/Euler2D.okl",
  //                                             "euler_vol_2d", info);
  app->vol = occaDeviceBuildKernelFromSource(app->device, "okl/Euler2D.okl",
  					     "euler_vol_2d_curved", info);
  app->surf = occaDeviceBuildKernelFromSource(app->device, "okl/Euler2D.okl",
                                              "euler_surf_2d", info);
  app->update = occaDeviceBuildKernelFromSource(app->device, "okl/Euler2D.okl",
                                                "euler_update_2d_curved", info);
  app->test = occaDeviceBuildKernelFromSource(app->device, "okl/Euler2D.okl",
                                              "test_kernel_2d", info);
#else
  printf("building 3D kernels\n");
  
#ifdef TESTING
  app->vol_sub1 = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl", "euler_vol_3d_part1", info);

  app->vol_sub2 = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl", "euler_vol_3d_part2", info);

  app->vol = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl","euler_vol_3d_testing", info);

  app->update = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl","euler_update_3d_curved_testing", info);

  app->surf = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl","euler_surf_3d", info);

  app->logmean = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl","euler_logmean_testing", info);
#else
  app->vol = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl","euler_vol_3d", info);

  app->update = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl","euler_update_3d_curved", info);

  app->surf = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl","euler_surf_3d", info);
#endif
/*  
  app->vol = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3Daffine.okl", "euler_vol_3d", info);
  app->update = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3Daffine.okl","euler_update_3d", info);
  app->surf = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3Daffine.okl","euler_surf_3d", info);
*/
  app->test = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl",
                                              "test_kernel", info);
  app->aux = occaDeviceBuildKernelFromSource(app->device, "okl/Euler3D.okl",
                                              "euler_3d_compute_aux", info);

#endif
  printf("built kernels!\n");

  return app;
}

static void modify_mapP(app_t *app, int usePeriodic)
{

  const uintloc_t E = app->hm->E;
  // const int Nq = app->hops->Nq;
  const int Nfq = app->hops->Nfq;
  const int Nfaces = app->hops->Nfaces;
  // const int Nfgeo = app->hops->Nfgeo;

  if (usePeriodic == 1)
  {
    // assume 2D square, find min/max x and y
    double vxmin = 1e9;
    double vymin = 1e9;
    double vxmax = -1e9;
    double vymax = -1e9;
#if VDIM == 3
    double vzmin = 1e9;
    double vzmax = -1e9;
#endif

    // find min/max x,y,z values
    for (uintloc_t e = 0; e < E; ++e)
    {
      for (int i = 0; i < NVERTS; ++i)
      {
        double vx = app->hm->EToVX[NVERTS * VDIM * e + VDIM * i + 0];
        double vy = app->hm->EToVX[NVERTS * VDIM * e + VDIM * i + 1];
        vxmin = fmin(vxmin, vx);
        vxmax = fmax(vxmax, vx);
        vymin = fmin(vymin, vy);
        vymax = fmax(vymax, vy);
#if VDIM == 3
        double vz = app->hm->EToVX[NVERTS * VDIM * e + VDIM * i + 2];
        vzmin = fmin(vzmin, vz);
        vzmax = fmax(vzmax, vz);
#endif
      }
    }
#if VDIM == 2
    printf("vx max/min = [%f,%f], vy max/min = [%f, %f]\n", vxmin, vxmax, vymin,
           vymax);
#else
    printf(
        "vx max/min = [%f,%f], vy max/min = [%f, %f], vz max/min = [%f, %f]\n",
        vxmin, vxmax, vymin, vymax, vzmin, vzmax);
#endif

    // find boundary nodes
    dfloat_t *xB = asd_malloc_aligned(sizeof(dfloat_t) * Nfq * Nfaces * E);
    dfloat_t *yB = asd_malloc_aligned(sizeof(dfloat_t) * Nfq * Nfaces * E);
#if VDIM == 3
    dfloat_t *zB = asd_malloc_aligned(sizeof(dfloat_t) * Nfq * Nfaces * E);
#endif
    int *idB = asd_malloc_aligned(sizeof(int) * Nfq * Nfaces * E);
    int sk = 0;
    for (uintloc_t e = 0; e < E; ++e)
    {
      for (int i = 0; i < Nfq * Nfaces; ++i)
      {
        int idM = i + Nfq * Nfaces * e;
        int idP = app->hops->mapPq[i + e * Nfq * Nfaces];
        if (idM == idP)
        {
          xB[sk] = app->hops->xyzf[i + 0 * Nfq * Nfaces + e * Nfq * Nfaces * 3];
          yB[sk] = app->hops->xyzf[i + 1 * Nfq * Nfaces + e * Nfq * Nfaces * 3];
#if VDIM == 3
          zB[sk] = app->hops->xyzf[i + 2 * Nfq * Nfaces + e * Nfq * Nfaces * 3];
// printf("xb(%d) = %f; yb(%d) = %f; zb(%d) =
// %f\n",sk+1,xB[sk],sk+1,yB[sk],sk+1,zB[sk]);
#endif
          idB[sk] = idM;
          ++sk;
        }
      }
    }

    // find periodic node matches
    int xcnt = 0, ycnt = 0, zcnt = 0;
    double tol = 1e-5;
    int num_boundary_nodes = sk;
    printf("num boundary nodes = %d\n", num_boundary_nodes);
    for (int i = 0; i < num_boundary_nodes; ++i)
    {
      dfloat_t xi = xB[i];
      dfloat_t yi = yB[i];
#if VDIM == 3
      dfloat_t zi = zB[i];
#endif

      for (int j = 0; j < num_boundary_nodes; ++j)
      {
        dfloat_t xj = xB[j];
        dfloat_t yj = yB[j];
#if VDIM == 3
        dfloat_t zj = zB[j];
#endif

        if (i != j)
        {
#if VDIM == 2
          // if match on x faces
          if ((fabs(xi - vxmin) < tol || fabs(xi - vxmax) < tol) &&
              (fabs(yi - yj) < tol))
          {
            app->hops->mapPq[idB[i]] = idB[j];
          }
          // if match on y faces
          if ((fabs(yi - vymin) < tol || fabs(yi - vymax) < tol) &&
              (fabs(xi - xj) < tol))
          {
            app->hops->mapPq[idB[i]] = idB[j];
          }
#else
          // if on
          int on_xmin = fabs(xi - vxmin) < tol;
          int on_ymin = fabs(yi - vymin) < tol;
          int on_zmin = fabs(zi - vzmin) < tol;
          int on_xmax = fabs(xi - vxmax) < tol;
          int on_ymax = fabs(yi - vymax) < tol;
          int on_zmax = fabs(zi - vzmax) < tol;

          // match on x face
          if ((on_xmin || on_xmax) && (fabs(yi - yj) < tol) &&
              (fabs(zi - zj) < tol))
          {
            app->hops->mapPq[idB[i]] = idB[j];
            xcnt++;
          }
          // match on y face
          if ((on_ymin || on_ymax) && (fabs(xi - xj) < tol) &&
              (fabs(zi - zj) < tol))
          {
            app->hops->mapPq[idB[i]] = idB[j];
            ycnt++;
          }
          // match on z face
          if ((on_zmin || on_zmax) && (fabs(xi - xj) < tol) &&
              (fabs(yi - yj) < tol))
          {
            app->hops->mapPq[idB[i]] = idB[j];
            zcnt++;
          }
#endif
        }
      }
    }
    printf("# of xbpts = %d, # of ybpts = %d, # of zbpts = %d\n", xcnt, ycnt,
           zcnt);

  } // end periodic stuff

  // FIX
  // modify mapPq to accomodate arrays of size NFIELDS
  uintloc_t* mapPqFields = (uintloc_t *)malloc(sizeof(uintloc_t) * Nfq*Nfaces*E);

  for (uintloc_t e = 0; e < E; ++e)
  {
    for (int i = 0; i < Nfq * Nfaces; ++i)
    {
      int idP = app->hops->mapPq[i + e * Nfq * Nfaces];
      int enbr = idP / (Nfq * Nfaces);
      //app->hops->mapPq[i + e * Nfq * Nfaces] =
      //	(idP - Nfq * Nfaces * enbr) + enbr * Nfq * Nfaces * NFIELDS;
      mapPqFields[i + e * Nfq * Nfaces] = (idP - Nfq * Nfaces * enbr) + enbr * Nfq * Nfaces * NFIELDS;
    }
  }

  occaMemoryFree(app->mapPq);
  app->mapPq = device_malloc(app->device, sizeof(uintloc_t) * Nfq * Nfaces * E, mapPqFields);
  //app->mapPq = device_malloc(app->device, sizeof(uintloc_t) * Nfq * Nfaces * E,app->hops->mapPq);
}

static double get_hmin(app_t *app)
{

  const uintloc_t E = app->hm->E;
  const int Nq = app->hops->Nq;
  const int Nfq = app->hops->Nfq;
  const int Nfaces = app->hops->Nfaces;
  const int Nfgeo = app->hops->Nfgeo;

  double hmin = 1e9;
  for (uintloc_t e = 0; e < E; ++e)
  {
    double Jmin = 1e9;
    for (int i = 0; i < Nq; ++i)
    {
      Jmin = fmin(Jmin, app->hops->Jq[i + Nq * e]);
    }

    double sJmax = 0.0;
    for (int i = 0; i < Nfq * Nfaces; ++i)
    {
#if VDIM == 2
      sJmax = fmax(
          sJmax,
          app->hops->fgeo[i + 2 * Nfq * Nfaces + e * Nfq * Nfaces * Nfgeo]);
// printf("i = %d, e = %d: sJmax = %f\n",i,e,sJmax);
#else
      sJmax = fmax(
          sJmax,
          app->hops->fgeo[i + 3 * Nfq * Nfaces + e * Nfq * Nfaces * Nfgeo]);
#endif
    }
    hmin = fmin(hmin, Jmin / sJmax);
  }

  return hmin;
}

// convert conservative to entropy vars
static void VU(app_t *app, euler_fields U, euler_fields *V)
{
  // compute entropy vars
  dfloat_t rho = U.U1;
  dfloat_t rhou = U.U2;
  dfloat_t rhov = U.U3;
#if VDIM == 2
  dfloat_t E = U.U4;
#else
  dfloat_t rhow = U.U4;
  dfloat_t E = U.U5;
#endif

  dfloat_t gamma = app->prefs->physical_gamma;
#if VDIM == 2
  dfloat_t rhoe = E - .5f * (rhou * rhou + rhov * rhov) / rho;
#else
  dfloat_t rhoe = E - .5f * (rhou * rhou + rhov * rhov + rhow * rhow) / rho;
#endif
  dfloat_t s = LOGDF((gamma - 1.0) * rhoe / POWDF(rho, gamma));

  V->U1 = (-E) / rhoe + (gamma + 1.0 - s);
  V->U2 = rhou / rhoe;
  V->U3 = rhov / rhoe;
#if VDIM == 2
  V->U4 = (-rho) / rhoe;
#else
  V->U4 = rhow / rhoe;
  V->U5 = (-rho) / rhoe;
#endif
}

// convert entropy to conservative vars
static void UV(app_t *app, euler_fields V, euler_fields *U)
{
  dfloat_t V1 = V.U1;
  dfloat_t V2 = V.U2;
  dfloat_t V3 = V.U3;
  dfloat_t V4 = V.U4;
  dfloat_t gamma = app->prefs->physical_gamma;

#if VDIM == 2
  dfloat_t s = gamma - V1 + (V2 * V2 + V3 * V3) / (2.0 * V4);
  dfloat_t rhoe =
      POWDF((gamma - 1.0) / POWDF(-V4, gamma), 1.0 / (gamma - 1.0)) *
      EXPDF(-s / (gamma - 1.0));
  U->U1 = rhoe * (-V4);
  U->U2 = rhoe * (V2);
  U->U3 = rhoe * (V3);
  U->U4 = rhoe * (1.0 - (V2 * V2 + V3 * V3) / (2.0 * V4));
#else
  dfloat_t V5 = V.U5;
  dfloat_t s = gamma - V1 + (V2 * V2 + V3 * V3 + V4 * V4) / (2.0 * V5);
  dfloat_t rhoe =
      POWDF((gamma - 1.0) / POWDF(-V5, gamma), 1.0 / (gamma - 1.0)) *
      EXPDF(-s / (gamma - 1.0));
  U->U1 = rhoe * (-V5);
  U->U2 = rhoe * (V2);
  U->U3 = rhoe * (V3);
  U->U4 = rhoe * (V4);
  U->U5 = rhoe * (1.0 - (V2 * V2 + V3 * V3 + V4 * V4) / (2.0 * V5));
#endif
}

void euler_vortex(app_t *app, coord X, dfloat_t t, euler_fields *U)
{

  dfloat_t x = X.x;
  dfloat_t y = X.y;
  // dfloat_t z = X.z;
  dfloat_t gamma = app->prefs->physical_gamma;
  dfloat_t gm1 = (gamma - 1.0);

#if VDIM == 2
  // 2D vortex solution
  dfloat_t x0 = 5.0;
  dfloat_t y0 = 0.0;
  dfloat_t beta = 5.0;
  dfloat_t r2 = (x - x0 - t) * (x - x0 - t) + (y - y0) * (y - y0);

  dfloat_t u = 1.0 - beta * EXPDF(1.0 - r2) * (y - y0) / (2.0 * M_PI);
  dfloat_t v = beta * EXPDF(1.0 - r2) * (x - x0 - t) / (2.0 * M_PI);
  dfloat_t rho0 = 1.0 -
                  .5 * gm1 * (1.0 / (8.0 * gamma * M_PI * M_PI)) *
                      POWDF(beta * EXPDF(1.0 - r2), 2.0);
  dfloat_t rho = POWDF(rho0, (1.0 / gm1));
  dfloat_t rhou = (rho)*u;
  dfloat_t rhov = (rho)*v;
  dfloat_t p = POWDF((rho), gamma);
  dfloat_t E = p / gm1 + .5 * (rho) * (u * u + v * v);

  // const sol for testing
  
  rho = 1.0;
  rhou = 2.0;
  rhov = 5.0;
  E = 1.0 + .5f * (rhou * rhou + rhov * rhov) / rho;
  

  U->U1 = rho;
  U->U2 = rhou;
  U->U3 = rhov;
  U->U4 = E;

#else

  dfloat_t z = X.z;

  // 3D vortex on [0,10] x [0,20] x [0,10]
  dfloat_t x0 = 5.0;
  dfloat_t y0 = 5.0;
  // dfloat_t z0 = 5.0;
  dfloat_t xt = x - x0;
  dfloat_t yt = y - y0 - t;
  // dfloat_t zt = z - z0;

  // cross(X,[0,0,1]) = [-y,x,0]
  dfloat_t rx = -yt;
  dfloat_t ry = xt;
  dfloat_t rz = 0.0;
  dfloat_t r2 = rx * rx + ry * ry + rz * rz;

  dfloat_t rho0 = 1.0;
  dfloat_t p0 = 1.0 / gamma;
  dfloat_t Lmax = .4;

  dfloat_t L = Lmax * EXPDF(.5 * (1.0 - r2));
  dfloat_t tmp = POWDF(1.0 - .5 * gm1 * L * L, 1.0 / gm1);
  dfloat_t rho = rho0 * tmp;
  dfloat_t rhou = rho * (0.0 + rx * L);
  dfloat_t rhov = rho * (1.0 + ry * L);
  dfloat_t rhow = rho * (0.0 + rz * L);
  dfloat_t E = p0 / gm1 * (1.0 + POWDF(tmp, gamma)) +
               .5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rho;

  // const sol for testing
  
  rho = 1.0;
  rhou = 2.0;
  rhov = 3.0;
  rhow = 4.0;
  E = 1.0 + .5*(rhou*rhou+rhov*rhov+rhow*rhow)/rho;
  

  // more testing: entropy RHS
  /*
  dfloat_t du = EXPDF(-4.0*((x-5.0)*(x-5.0) + (y-10.0)*(y-10.0) + (z-5.0)*(z-5.0)));
  rho = 1.0 + du;
  dfloat_t u = du;
  rhou = rho*u;
  rhov = 0.0;
  rhow = 0.0;
  dfloat_t p = 1.0;
  E = p/gm1 + .5*rho*(u*u);
  */

  U->U1 = rho;
  U->U2 = rhou;
  U->U3 = rhov;
  U->U4 = rhow;
  U->U5 = E;
#endif
}

// posed on [-pi,pi]^3 in 3D
void euler_Taylor_Green(app_t *app, coord X, euler_fields *U)
{

  dfloat_t x = X.x;
  dfloat_t y = X.y;
  dfloat_t z = X.z;
  dfloat_t gamma = app->prefs->physical_gamma;
  dfloat_t gm1 = (gamma - 1.0);

  dfloat_t rho = 1.0;
  dfloat_t u = SINDF(x)*COSDF(y)*COSDF(z);
  dfloat_t v = -COSDF(x)*SINDF(y)*COSDF(z);
  dfloat_t w = 0.0;
  dfloat_t p = 100.0/gamma + (1.0/16.0)*(COSDF(2.0*x) + COSDF(2.0*y))*(2.0 + COSDF(2.0*z));

  /*
  //rho = 1.0;  u = 1.0;  v = 0.0;  p = 1.0; //testing
  rho = 2.0 + .1*((dfloat_t) (rand()) / (dfloat_t) (RAND_MAX));
  u   = 1.0 + .1*((dfloat_t) (rand()) / (dfloat_t) (RAND_MAX));
  v   = 1.0 + .1*((dfloat_t) (rand()) / (dfloat_t) (RAND_MAX));
  w = 1.0 + .1*((dfloat_t) (rand()) / (dfloat_t) (RAND_MAX));
  p = 2.0 + .1*((dfloat_t) (rand()) / (dfloat_t) (RAND_MAX));
  */
  U->U1 = rho;
  U->U2 = rho*u;
  U->U3 = rho*v;
  U->U4 = rho*w;
  // p = (gamma-1)*internal energy = (gm1)*(E-.5*rho*(u^2+v^2+w^2));
  // E = p/(gamma-1) + .5*rho*||u||^2
  U->U5 = p/gm1 + .5*rho*(u*u+v*v+w*w);
}

#if 0
static void app_test(app_t *app)
{

  printf("Testing app...\n");

  occaKernelRun(app->test, occaInt(app->hm->E), app->Q, app->Qf, app->rhsQ,
                app->rhsQf);
}
#endif

static void test_rhs(app_t *app)
{
  occaKernelRun(app->vol, occaInt(app->hm->E), app->vgeo, app->vfgeo, app->nrJ, app->nsJ,
  		app->ntJ, app->Drq, app->Dsq, app->Dtq, app->Drstq, app->VqLq, app->VfPq,
  		app->Q, app->Qf, app->rhsQ, app->rhsQf);

  occaKernelRun(app->surf, occaInt(app->hm->E), app->fgeo, app->mapPq,
  		app->VqLq, app->Qf, app->rhsQf, app->rhsQ);
  int K = app->hm->E;
  int Nq = app->hops->Nq;
  int Nfq = app->hops->Nfq;
  int Nfaces = app->hops->Nfaces;
  int size = NFIELDS * Nq * K;
  int sizef = NFIELDS * Nfq * Nfaces * K;
  const int Nvgeo = app->hops->Nvgeo;
  const int Nfgeo = app->hops->Nfgeo;

#if 0

  for (int e = 0; e < K; ++e){
    for (int i = 0; i < Nq; ++i){
      dfloat_t xq = app->hops->xyzq[i + 0*Nq + e*Nq*3];
      dfloat_t yq = app->hops->xyzq[i + 1*Nq + e*Nq*3];
      dfloat_t zq = app->hops->xyzq[i + 2*Nq + e*Nq*3];
      printf("xq(%d,%d) = %f;yq(%d,%d) = %f;zq(%d,%d) = %f;\n",i+1,e+1,xq,i+1,e+1,yq,i+1,e+1,zq);

      dfloat_t rxJ = app->hops->vgeo[e * Nq * Nvgeo + 0 * Nq + i];
      dfloat_t ryJ = app->hops->vgeo[e * Nq * Nvgeo + 1 * Nq + i];
      dfloat_t rzJ = app->hops->vgeo[e * Nq * Nvgeo + 2 * Nq + i];
      dfloat_t sxJ = app->hops->vgeo[e * Nq * Nvgeo + 3 * Nq + i];
      dfloat_t syJ = app->hops->vgeo[e * Nq * Nvgeo + 4 * Nq + i];
      dfloat_t szJ = app->hops->vgeo[e * Nq * Nvgeo + 5 * Nq + i];
      dfloat_t txJ = app->hops->vgeo[e * Nq * Nvgeo + 6 * Nq + i];
      dfloat_t tyJ = app->hops->vgeo[e * Nq * Nvgeo + 7 * Nq + i];
      dfloat_t tzJ = app->hops->vgeo[e * Nq * Nvgeo + 8 * Nq + i];

      printf("rxJ(%d,%d) = %f; sxJ(%d,%d) = %f; txJ(%d,%d) = %f;\n",i+1,e+1,rxJ,i+1,e+1,sxJ,i+1,e+1,txJ);
      printf("ryJ(%d,%d) = %f; syJ(%d,%d) = %f; tyJ(%d,%d) = %f;\n",i+1,e+1,ryJ,i+1,e+1,syJ,i+1,e+1,tyJ);
      printf("rzJ(%d,%d) = %f; szJ(%d,%d) = %f; tzJ(%d,%d) = %f;\n",i+1,e+1,rzJ,i+1,e+1,szJ,i+1,e+1,tzJ);
    }
    for (int i = 0; i < Nfq*Nfaces; ++i){
      dfloat_t xf = app->hops->xyzf[i + 0*Nfq*Nfaces + e*Nfq*Nfaces*3];
      dfloat_t yf = app->hops->xyzf[i + 1*Nfq*Nfaces + e*Nfq*Nfaces*3];
      dfloat_t zf = app->hops->xyzf[i + 2*Nfq*Nfaces + e*Nfq*Nfaces*3];
      printf("xf(%d,%d) = %f;yf(%d,%d) = %f;zf(%d,%d) = %f;\n",i+1,e+1,xf,i+1,e+1,yf,i+1,e+1,zf);

      dfloat_t rxJ = app->hops->vfgeo[e * Nfq*Nfaces * Nvgeo + 0 * Nfq*Nfaces + i];
      dfloat_t ryJ = app->hops->vfgeo[e * Nfq*Nfaces * Nvgeo + 1 * Nfq*Nfaces + i];
      dfloat_t rzJ = app->hops->vfgeo[e * Nfq*Nfaces * Nvgeo + 2 * Nfq*Nfaces + i];
      dfloat_t sxJ = app->hops->vfgeo[e * Nfq*Nfaces * Nvgeo + 3 * Nfq*Nfaces + i];
      dfloat_t syJ = app->hops->vfgeo[e * Nfq*Nfaces * Nvgeo + 4 * Nfq*Nfaces + i];
      dfloat_t szJ = app->hops->vfgeo[e * Nfq*Nfaces * Nvgeo + 5 * Nfq*Nfaces + i];
      dfloat_t txJ = app->hops->vfgeo[e * Nfq*Nfaces * Nvgeo + 6 * Nfq*Nfaces + i];
      dfloat_t tyJ = app->hops->vfgeo[e * Nfq*Nfaces * Nvgeo + 7 * Nfq*Nfaces + i];
      dfloat_t tzJ = app->hops->vfgeo[e * Nfq*Nfaces * Nvgeo + 8 * Nfq*Nfaces + i];

      printf("rxJf(%d,%d) = %f; sxJf(%d,%d) = %f; txJf(%d,%d) = %f;\n",i+1,e+1,rxJ,i+1,e+1,sxJ,i+1,e+1,txJ);
      printf("ryJf(%d,%d) = %f; syJf(%d,%d) = %f; tyJf(%d,%d) = %f;\n",i+1,e+1,ryJ,i+1,e+1,syJ,i+1,e+1,tyJ);
      printf("rzJf(%d,%d) = %f; szJf(%d,%d) = %f; tzJf(%d,%d) = %f;\n",i+1,e+1,rzJ,i+1,e+1,szJ,i+1,e+1,tzJ);

      dfloat_t nxJ = app->hops->fgeo[e * Nfq*Nfaces * Nfgeo + 0 * Nfq*Nfaces + i];
      dfloat_t nyJ = app->hops->fgeo[e * Nfq*Nfaces * Nfgeo + 1 * Nfq*Nfaces + i];
      dfloat_t nzJ = app->hops->fgeo[e * Nfq*Nfaces * Nfgeo + 2 * Nfq*Nfaces + i];

      printf("nxJ(%d,%d) = %f;nyJ(%d,%d) = %f;nzJ(%d,%d) = %f;\n",i+1,e+1,nxJ,i+1,e+1,nyJ,i+1,e+1,nzJ);

      int idM = i + e*Nfq*Nfaces;
      //int idP = app->hops->mapPqNoFields[i + e*Nfq*Nfaces];
      int idP = app->hops->mapPq[idM];
      printf("mapM(%d,%d) = %d; mapP(%d,%d) = %d;\n",i+1,e+1,idM+1,i+1,e+1,idP+1);
    }
  }
#endif

#if 0
  // checking node maps which don't match
  int i = 0;
  int e = 0;
  int idM = i + e*Nfq*Nfaces;
  int idP = app->hops->mapPq[idM];
  printf("idM = %d, idP = %d\n",idM,idP);

  dfloat_t xf = app->hops->xyzf[i + 0*Nfq*Nfaces + e*Nfq*Nfaces*3];
  dfloat_t yf = app->hops->xyzf[i + 1*Nfq*Nfaces + e*Nfq*Nfaces*3];
  dfloat_t zf = app->hops->xyzf[i + 2*Nfq*Nfaces + e*Nfq*Nfaces*3];
  printf("xyzfM = %f, %f, %f\n",xf,yf,zf);

  i = 9;
  e = 88;
  printf("idM2 = %d\n",i+e*Nfq*Nfaces);
  xf = app->hops->xyzf[i + 0*Nfq*Nfaces + e*Nfq*Nfaces*3];
  yf = app->hops->xyzf[i + 1*Nfq*Nfaces + e*Nfq*Nfaces*3];
  zf = app->hops->xyzf[i + 2*Nfq*Nfaces + e*Nfq*Nfaces*3];
  printf("xyzfP = %f, %f, %f\n",xf,yf,zf);
#endif

#if 1 // test RHS with entropy vars after computing vol/surface RHS
  dfloat_t *Q = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * size);
  occaCopyMemToPtr(Q, app->Q, size * sizeof(dfloat_t), occaNoOffset);
  dfloat_t *rhs = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * size);
  occaCopyMemToPtr(rhs, app->rhsQ, size * sizeof(dfloat_t), occaNoOffset);

  dfloat_t *rhsf = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * sizef);
  occaCopyMemToPtr(rhsf, app->rhsQf, sizef * sizeof(dfloat_t), occaNoOffset);
  //  occaDeviceFinish(app->device);

  dfloat_t Srhs = 0.0;
  dfloat_t rhssum = 0.0;
  dfloat_t rhsfsum = 0.0;
  for (int e = 0; e < K; ++e){
    dfloat_t *VV = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nq * NFIELDS);
    dfloat_t *rr = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nq * NFIELDS);
    for (int i = 0; i < Nq; ++i){
      int id = i + NFIELDS*Nq*e;

      euler_fields U, V;
      U.U1 = Q[id + 0*Nq];;
      U.U2 = Q[id + 1*Nq];;
      U.U3 = Q[id + 2*Nq];;
      U.U4 = Q[id + 3*Nq];;
      U.U5 = Q[id + 4*Nq];;

      VU(app, U, &V);
      VV[i       ] = V.U1;
      VV[i + Nq  ] = V.U2;
      VV[i + 2*Nq] = V.U3;
      VV[i + 3*Nq] = V.U4;
      VV[i + 4*Nq] = V.U5;

      rr[i     ] = rhs[id];
      rr[i + Nq] = rhs[id + Nq];
      rr[i + 2*Nq] = rhs[id + 2*Nq];
      rr[i + 3*Nq] = rhs[id + 3*Nq];
      rr[i + 4*Nq] = rhs[id + 4*Nq];
    }

    // check testing projected RHS with projected entropy vars (should be zero for affine)
    for (int i = 0; i < Nq; ++i){
      dfloat_t V1 = 0.0;      dfloat_t V2 = 0.0;      dfloat_t V3 = 0.0;
      dfloat_t V4 = 0.0;      dfloat_t V5 = 0.0;

      dfloat_t r1 = 0.0;      dfloat_t r2 = 0.0;      dfloat_t r3 = 0.0;
      dfloat_t r4 = 0.0;      dfloat_t r5 = 0.0;

      for (int j = 0; j < Nq; ++j){
        dfloat_t VqPq_ij = app->hops->VqPq[i + j * Nq];
        V1 += VqPq_ij*VV[j + 0*Nq];
        V2 += VqPq_ij*VV[j + 1*Nq];
        V3 += VqPq_ij*VV[j + 2*Nq];
        V4 += VqPq_ij*VV[j + 3*Nq];
        V5 += VqPq_ij*VV[j + 4*Nq];

	r1 += VqPq_ij*rr[j + 0*Nq];
	r2 += VqPq_ij*rr[j + 1*Nq];
	r3 += VqPq_ij*rr[j + 2*Nq];
	r4 += VqPq_ij*rr[j + 3*Nq];
	r5 += VqPq_ij*rr[j + 4*Nq];
      }
      dfloat_t wq = (app->hops->wq[i]);
      dfloat_t Jq =  (app->hops->Jq[i + e * Nq]);

      /*
	r1 = rr[i+0*Nq];
	r2 = rr[i+1*Nq];
	r3 = rr[i+2*Nq];
	r4 = rr[i+3*Nq];
	r5 = rr[i+4*Nq];
      */

      Srhs += wq*(r1+r2+r3+r4+r5); //wq*(V1*r1 + V2*r2 + V3*r3 + V4*r4 + V5*r5);
      rhssum += rhs[i + 0*Nq + Nq*NFIELDS*e];
      //Srhs += V1;
      //printf("rhoq(%d,%d) = %f;\n",i+1,e+1,Q[i + Nq*NFIELDS*e]);
    }

    for (int i = 0; i < Nfq*Nfaces; ++i){
      rhsfsum += rhsf[i + 0*Nfq*Nfaces + Nfq*Nfaces*NFIELDS*e];//wq*(V1*r1);
      //printf("FxS(%d,%d) = %f;\n",i+1,e+1,rhsf[i + Nfq*Nfaces*NFIELDS*e]);
    }
  }
  printf("Rhs vol = %g, rhs face = %g, entropy rhs = %g\n",rhssum,rhsfsum,Srhs);
  //printf("Entropy rhs vol = %g\n",Srhs);
#endif

}

static void rk_step(app_t *app, double rka, double rkb, double dt)
{

#if VDIM == 2

  occaKernelRun(app->vol, occaInt(app->hm->E), app->vgeo, app->vfgeo, app->nrJ, app->nsJ,
                app->Drq, app->Dsq, app->VqLq, app->VfPq, app->Q, app->Qf,
                app->rhsQ, app->rhsQf);

  occaKernelRun(app->surf, occaInt(app->hm->E), app->fgeo, app->mapPq,
                app->VqLq, app->Qf, app->rhsQf, app->rhsQ);

  occaKernelRun(app->update, occaInt(app->hm->E), app->Jq, app->VqPq, app->VfPq,
                occaDfloat((dfloat_t)rka), occaDfloat((dfloat_t)rkb),
                occaDfloat((dfloat_t)dt), app->rhsQ, app->resQ, app->Q,
                app->Qf);

#else


#ifdef TESTING
//  occaKernelRun(app->logmean, occaInt(app->hm->E), app->vgeo, app->vfgeo, app->nrJ, app->nsJ,
//                app->ntJ, app->Drq, app->Dsq, app->Dtq, app->Drstq, app->VqLq, app->VfPq,
//                app->Q, app->Qf, app->rhsQ, app->rhsQf, app->rhoLog, app->betaLog, app->storage);

  occaKernelRun(app->vol_sub1, occaInt(app->hm->E), app->vgeo, app->vfgeo, app->nrJ, app->nsJ,
                app->ntJ, app->Drq, app->Dsq, app->Dtq, app->Drstq, app->VqLq, app->VfPq,
                app->Q, app->Qf, app->rhsQ, app->rhsQf, app->rhoLog, app->betaLog, app->storage, app->store_pNq, app->store_pNfqNfaces);

  occaKernelRun(app->vol_sub2, occaInt(app->hm->E), app->vgeo, app->vfgeo, app->nrJ, app->nsJ,
                app->ntJ, app->Drq, app->Dsq, app->Dtq, app->Drstq, app->VqLq, app->VfPq,
                app->Q, app->Qf, app->rhsQ, app->rhsQf, app->rhoLog, app->betaLog, app->storage, app->store_pNq, app->store_pNfqNfaces);

/*
  occaKernelRun(app->vol, occaInt(app->hm->E), app->vgeo, app->vfgeo, app->nrJ, app->nsJ,
                app->ntJ, app->Drq, app->Dsq, app->Dtq, app->Drstq, app->VqLq, app->VfPq,
                app->Q, app->Qf, app->rhsQ, app->rhsQf, app->rhoLog, app->betaLog, app->storage);
*/
  occaKernelRun(app->surf, occaInt(app->hm->E), app->fgeo, app->mapPq,
                app->VqLq, app->Qf, app->rhsQf, app->rhsQ);

  occaKernelRun(app->update, occaInt(app->hm->E), app->Jq, app->VqPq, app->VfPq,
                occaDfloat((dfloat_t)rka), occaDfloat((dfloat_t)rkb),
                occaDfloat((dfloat_t)dt), app->rhsQ, app->resQ, app->Q,
                app->Qf, app->rhoLog, app->betaLog);
#else
  occaKernelRun(app->vol, occaInt(app->hm->E), app->vgeo, app->vfgeo, app->nrJ, app->nsJ,
                app->ntJ, app->Drq, app->Dsq, app->Dtq, app->Drstq, app->VqLq, app->VfPq,
                app->Q, app->Qf, app->rhsQ, app->rhsQf);

  occaKernelRun(app->surf, occaInt(app->hm->E), app->fgeo, app->mapPq,
                app->VqLq, app->Qf, app->rhsQf, app->rhsQ);

  occaKernelRun(app->update, occaInt(app->hm->E), app->Jq, app->VqPq, app->VfPq,
                occaDfloat((dfloat_t)rka), occaDfloat((dfloat_t)rkb),
                occaDfloat((dfloat_t)dt), app->rhsQ, app->resQ, app->Q,
                app->Qf);
#endif

#endif
}

static void rk_run(app_t *app, double dt, double FinalTime)
{
  printf("Running...\n");

  int Nsteps = (int)ceil(FinalTime / dt);
  int interval = (Nsteps / 10);
  if (Nsteps < 10)
  {
    interval = 1;
  }
  dt = (double)FinalTime / Nsteps;

  // alloc for kinetic energy computation
  int K = app->hm->E;
  dfloat_t *KE_time = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nsteps);
  dfloat_t *KE = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * K);
  occaMemory o_KE = device_malloc(app->device, sizeof(dfloat_t) * K, NULL);


  for (int tstep = 0; tstep < Nsteps; ++tstep)
  {
    for (int INTRK = 0; INTRK < 5; ++INTRK)
    {
      const double rka = app->rk4a[INTRK];
      const double rkb = app->rk4b[INTRK];
      rk_step(app, rka, rkb, dt);
      // app_test(app);
    }

#if 0
    occaKernelRun(app->aux, occaInt(app->hm->E), app->Jq, app->wq, app->Q, o_KE);
    occaCopyMemToPtr(KE, o_KE, K * sizeof(dfloat_t), occaNoOffset);
    double ke = 0.0;
    for (int e = 0; e < K; ++e){
      ke += KE[e];
    }
    KE_time[tstep] = ke;
    //printf("kinetic energy at step %d = %f\n",tstep,ke);
#endif
    if (tstep % interval == 0)
    {
      occaDeviceFinish(app->device);
      //printf("on timestep %d out of %d\n", tstep, Nsteps);
      printf("on timestep %d out of %d: ke = %f\n", tstep, Nsteps,KE_time[tstep]);

    }
  }

#if 0
  int sk = 1;
  if (Nsteps > 2500){
    sk = Nsteps/2500; // int division
    printf("sk = %d\n",sk);
  }
  printf("KE_time = [");
  for (int tstep = 0; tstep < Nsteps; tstep+=sk){
    printf("%f ",KE_time[tstep]);
  }
  printf("];\n");
  printf("t = [");
  for (int tstep = 0; tstep < Nsteps; tstep+=sk){
    printf("%f ",((double)tstep)*dt);
  }
  printf("];\n");
#endif

}

static void app_free(app_t *app)
{
  occaMemoryFree(app->vgeo);
  occaMemoryFree(app->fgeo);
  occaMemoryFree(app->Jq);

  occaMemoryFree(app->mapPq);
  occaMemoryFree(app->Fmask);

  occaMemoryFree(app->nrJ);
  occaMemoryFree(app->nsJ);

  occaMemoryFree(app->Drq);
  occaMemoryFree(app->Dsq);

#if VDIM == 3
  occaMemoryFree(app->ntJ);
  occaMemoryFree(app->Dtq);
#endif

  occaMemoryFree(app->Vq);
  occaMemoryFree(app->Pq);

  occaMemoryFree(app->VqLq);
  occaMemoryFree(app->VqPq);
  occaMemoryFree(app->VfPq);
  occaMemoryFree(app->Vfqf);

  occaMemoryFree(app->Q);
  occaMemoryFree(app->Qf);
  occaMemoryFree(app->rhsQ);
  occaMemoryFree(app->resQ);

#ifdef TESTING
  occaMemoryFree(app->rhoLog);
  occaMemoryFree(app->betaLog);
  occaMemoryFree(app->storage);
  occaMemoryFree(app->store_pNq);
  occaMemoryFree(app->store_pNfqNfaces);
#endif

  // free info
  occaKernelInfoFree(app->info);

  // free kernels
  //#if ELEM_TYPE == 0 // triangle
  occaKernelFree(app->test);
  occaKernelFree(app->vol);
  occaKernelFree(app->surf);
  occaKernelFree(app->update);
  //  occaKernelFree(app->face);
  //#endif

  prefs_free(app->prefs);
  asd_free(app->prefs);

  occaStreamFree(app->copy);
  occaStreamFree(app->cmdx);

  uintmax_t bytes = occaDeviceBytesAllocated(app->device);
  uintmax_t total_bytes = occaDeviceMemorySize(app->device);
  ASD_INFO("");
  ASD_INFO("Device bytes allocated %ju (%.2f GiB) out of %ju (%.2f GiB)", bytes,
           ((double)bytes) / GiB, total_bytes, ((double)total_bytes) / GiB);

  occaDeviceFree(app->device);

  host_mesh_free(app->hm);
  asd_free(app->hm);

  host_operators_free(app->hops);
  asd_free(app->hops);
}
// }}}

// {{{ Main
static void usage()
{
  const char *help_text =
      "  " APP_NAME " [options] prefs_file\n"
      "\n"
      "  there are four possible options to this program, some of which \n"
      "  have multiple names:\n"
      "\n"
      "    -h -? --help --HELP\n"
      "    -d --debug\n"
      "    -D --devices\n"
      "    -V --version\n"
      "    -v --verbose  (which may be repeated for more verbosity)\n"
      "\n";
  ASD_ROOT_INFO(help_text);
}

int main(int argc, char *argv[])
{
  int status = EXIT_SUCCESS;
  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;

  ASD_MPI_CHECK(MPI_Init(&argc, &argv));
  ASD_MPI_CHECK(MPI_Comm_rank(comm, &rank));

  //
  // parse command line
  //
  void *options = asd_gopt_sort(
      &argc, (const char **)argv,
      asd_gopt_start(asd_gopt_option('h', 0, asd_gopt_shorts('h', '?'),
                                     asd_gopt_longs("help", "HELP")),
                     asd_gopt_option('d', 0, asd_gopt_shorts('d'),
                                     asd_gopt_longs("debug")),
                     asd_gopt_option('D', 0, asd_gopt_shorts('D'),
                                     asd_gopt_longs("devices")),
                     asd_gopt_option('V', 0, asd_gopt_shorts('V'),
                                     asd_gopt_longs("version")),
                     asd_gopt_option('v', ASD_GOPT_REPEAT, asd_gopt_shorts('v'),
                                     asd_gopt_longs("verbose"))));

  if (asd_gopt(options, 'd'))
    debug(comm);

  if (asd_gopt(options, 'h'))
  {
    usage();
    goto finalize;
  }

  if (asd_gopt(options, 'V'))
    ASD_ROOT_INFO("app Version: %s", "unknown");

  if (asd_gopt(options, 'D'))
    occaPrintAvailableDevices();

  int verbosity = (int)asd_gopt(options, 'v');

  if (argc != 2)
  {
    ASD_LERROR("Unexpected number of arguments.");
    usage();
    status = EXIT_FAILURE;
    goto finalize;
  }

  //
  // initialize
  //
  init_libs(comm, verbosity);
  app_t *app = app_new(argv[1], comm);
  print_precision();

  int usePeriodic = 1;
  modify_mapP(app, usePeriodic);
  // return 0;

  //
  // run
  //

  // set initial condition
  const uintloc_t K = app->hm->E;
  const int Nq = app->hops->Nq;
  const int Nfq = app->hops->Nfq;
  const int Nfaces = app->hops->Nfaces;

#if 0
  // check maps
  for (uintloc_t e = 0; e < K; ++e){
    for (int i = 0; i < Nfq*Nfaces; ++i){
      dfloat_t x = app->hops->xyzf[i + 0*Nfq*Nfaces + e*Nfq*Nfaces*3];
      dfloat_t y = app->hops->xyzf[i + 1*Nfq*Nfaces + e*Nfq*Nfaces*3];
#if VDIM == 3
      dfloat_t z = app->hops->xyzf[i + 2*Nfq*Nfaces + e*Nfq*Nfaces*3];
#endif
      int idP = app->hops->mapPq[i + e*Nfq*Nfaces];

#if VDIM == 2
      printf("xf(%d,%d) = %f; yf(%d,%d) = %f; mapPq(%d,%d) = %d;\n",i+1,e+1,x,i+1,e+1,y,i+1,e+1,idP+1);
#else
      printf("xf(%d,%d) = %f; yf(%d,%d) = %f; zf(%d,%d) = %f; mapPq(%d,%d) = %d;\n",i+1,e+1,x,i+1,e+1,y,i+1,e+1,z,i+1,e+1,idP+1);
#endif

    }
  }
  return 0;
#endif

  dfloat_t *Q =
      (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nq * NFIELDS * K);
  dfloat_t *Qvq =
      (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nq * NFIELDS * K);
  dfloat_t *Qvf = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nfq *
                                                 Nfaces * NFIELDS * K);
  dfloat_t *resQ =
      (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nq * NFIELDS * K);

#ifdef TESTING
  dfloat_t *rhoLog = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nq * Nfq * Nfaces * app->prefs->kernel_KblkV * K);
  dfloat_t *betaLog = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nq * Nfq * Nfaces * app->prefs->kernel_KblkV * K);
  dfloat_t *storage = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nq * Nfq * Nfaces * app->prefs->kernel_KblkV * K * NFIELDS);
  dfloat_t *store_pNq = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * NFIELDS * Nq * Nfq * Nfaces * app->prefs->kernel_KblkV * K);
  dfloat_t *store_pNfqNfaces = (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * NFIELDS * Nfq * Nfaces * Nq * app->prefs->kernel_KblkV * K);
  for (int i=0; i<NFIELDS*K*Nq*Nfq*Nfaces*app->prefs->kernel_KblkV; ++i) {
    storage[i] = 0.0f;
  }
  for (int i=0; i<K*Nq*Nfq*Nfaces*app->prefs->kernel_KblkV; ++i) {
    rhoLog[i] = 0.0f;
    betaLog[i] = 0.0f;
  }
  for(int i=0; i<K*NFIELDS*Nq*Nfq*Nfaces*app->prefs->kernel_KblkV; ++i) {
    store_pNq[i] = 0.0f;
    store_pNfqNfaces[i] = 0.0f;
  }
#endif

  for (unsigned int i = 0; i < K * NFIELDS * Nq; ++i)
  {
    resQ[i] = 0.0;
  }
  for (uintloc_t e = 0; e < K; ++e)
  {
    for (int i = 0; i < Nq; ++i)
    {
      coord X;
      X.x = app->hops->xyzq[i + 0 * Nq + e * Nq * 3];
      X.y = app->hops->xyzq[i + 1 * Nq + e * Nq * 3];
#if VDIM == 3
      X.z = app->hops->xyzq[i + 2 * Nq + e * Nq * 3];
#endif

      // vortex solution, start at time t = 0
      euler_fields U;
      euler_vortex(app, X, 0.0, &U);
      //euler_Taylor_Green(app,X,&U);

      Q[i + 0 * Nq + e * Nq * NFIELDS] = U.U1;
      Q[i + 1 * Nq + e * Nq * NFIELDS] = U.U2;
      Q[i + 2 * Nq + e * Nq * NFIELDS] = U.U3;
      Q[i + 3 * Nq + e * Nq * NFIELDS] = U.U4;
#if VDIM == 3
      Q[i + 4 * Nq + e * Nq * NFIELDS] = U.U5;
#endif
    }

    // project Q onto polynomial basis
    dfloat_t *Qe =
        (dfloat_t *)asd_malloc_aligned(sizeof(dfloat_t) * Nq * NFIELDS);
    for (int fld = 0; fld < NFIELDS; ++fld)
    {
      for (int i = 0; i < Nq; ++i)
      {
        dfloat_t qi = 0.0;
        for (int j = 0; j < Nq; ++j)
        {
          dfloat_t Qj = Q[j + fld * Nq + e * Nq * NFIELDS];
          dfloat_t VqPq_ij = app->hops->VqPq[i + j * Nq];
          qi += VqPq_ij * Qj;
        }
        Qe[i] = qi;
      }
      // copy back projected init cond to global array
      for (int i = 0; i < Nq; ++i)
      {
        Q[i + fld * Nq + e * Nq * NFIELDS] = Qe[i];
      }
    }

    // also initialize rhsQ, rhsQf to u(VqPq*v) and u(VfPq*v)
    for (int i = 0; i < Nq; ++i)
    {
      euler_fields U, V;
      U.U1 = Q[i + 0 * Nq + e * Nq * NFIELDS];
      U.U2 = Q[i + 1 * Nq + e * Nq * NFIELDS];
      U.U3 = Q[i + 2 * Nq + e * Nq * NFIELDS];
      U.U4 = Q[i + 3 * Nq + e * Nq * NFIELDS];
#if VDIM == 3
      U.U5 = Q[i + 4 * Nq + e * Nq * NFIELDS];
#endif

      // compute entropy vars
      VU(app, U, &V);
      Qe[i + 0 * Nq] = V.U1;
      Qe[i + 1 * Nq] = V.U2;
      Qe[i + 2 * Nq] = V.U3;
      Qe[i + 3 * Nq] = V.U4;
#if VDIM == 3
      Qe[i + 4 * Nq] = V.U5;
#endif
    }
    // project entropy vars, eval at qpts
    for (int i = 0; i < Nq; ++i)
    {
      dfloat_t V1 = 0.0;
      dfloat_t V2 = 0.0;
      dfloat_t V3 = 0.0;
      dfloat_t V4 = 0.0;
#if VDIM == 3
      dfloat_t V5 = 0.0;
#endif
      for (int j = 0; j < Nq; ++j)
      {
        dfloat_t VqPq_ij = app->hops->VqPq[i + j * Nq];
        V1 += VqPq_ij * Qe[j + 0 * Nq];
        V2 += VqPq_ij * Qe[j + 1 * Nq];
        V3 += VqPq_ij * Qe[j + 2 * Nq];
        V4 += VqPq_ij * Qe[j + 3 * Nq];
#if VDIM == 3
        V5 += VqPq_ij * Qe[j + 4 * Nq];
#endif
      }
      euler_fields U, V;
      V.U1 = V1;
      V.U2 = V2;
      V.U3 = V3;
      V.U4 = V4;
#if VDIM == 3
      V.U5 = V5;
#endif
      UV(app, V, &U);

#ifdef COALESC
      Qvq[i*NFIELDS + e * Nq * NFIELDS] = U.U1;
      Qvq[1 + i*NFIELDS + e * Nq * NFIELDS] = U.U2;
      Qvq[2 + i*NFIELDS + e * Nq * NFIELDS] = U.U3;
      Qvq[3 + i*NFIELDS + e * Nq * NFIELDS] = U.U4;
      Qvq[4 + i*NFIELDS + e * Nq * NFIELDS] = U.U5;
#else
      Qvq[i + 0 * Nq + e * Nq * NFIELDS] = U.U1;
      Qvq[i + 1 * Nq + e * Nq * NFIELDS] = U.U2;
      Qvq[i + 2 * Nq + e * Nq * NFIELDS] = U.U3;
      Qvq[i + 3 * Nq + e * Nq * NFIELDS] = U.U4;
      Qvq[i + 4 * Nq + e * Nq * NFIELDS] = U.U5;
#endif
/*
#ifdef COALESC
      Qvq[i*NFIELDS + e * Nq * NFIELDS] = U.U1;
      Qvq[1 + i*NFIELDS + e * Nq * NFIELDS] = U.U2;
      Qvq[2 + i*NFIELDS + e * Nq * NFIELDS] = U.U3;
      Qvq[3 + i*NFIELDS + e * Nq * NFIELDS] = U.U4;
#if VDIM == 3      
      Qvq[4 + i*NFIELDS + e * Nq * NFIELDS] = U.U5;
#endif      
#else
      Qvq[i + 0 * Nq + e * Nq * NFIELDS] = U.U1;
      Qvq[i + 1 * Nq + e * Nq * NFIELDS] = U.U2;
      Qvq[i + 2 * Nq + e * Nq * NFIELDS] = U.U3;
      Qvq[i + 3 * Nq + e * Nq * NFIELDS] = U.U4;
#if VDIM == 3
      Qvq[i + 4 * Nq + e * Nq * NFIELDS] = U.U5;
#endif
#endif
*/
    }

    // project entropy vars, eval at face qpts
    for (int i = 0; i < Nfq * Nfaces; ++i)
    {
      dfloat_t V1 = 0.0;
      dfloat_t V2 = 0.0;
      dfloat_t V3 = 0.0;
      dfloat_t V4 = 0.0;
#if VDIM == 3
      dfloat_t V5 = 0.0;
#endif
      for (int j = 0; j < Nq; ++j)
      {
        dfloat_t VfPq_ij = app->hops->VfPq[i + j * Nfq * Nfaces];
        V1 += VfPq_ij * Qe[j + 0 * Nq];
        V2 += VfPq_ij * Qe[j + 1 * Nq];
        V3 += VfPq_ij * Qe[j + 2 * Nq];
        V4 += VfPq_ij * Qe[j + 3 * Nq];
#if VDIM == 3
        V5 += VfPq_ij * Qe[j + 4 * Nq];
#endif
      }

      euler_fields U, V;
      V.U1 = V1;
      V.U2 = V2;
      V.U3 = V3;
      V.U4 = V4;
#if VDIM == 3
      V.U5 = V5;
#endif
      UV(app, V, &U);
      
      Qvf[i + 0 * Nfq * Nfaces + e * Nfq * Nfaces * NFIELDS] = U.U1;
      Qvf[i + 1 * Nfq * Nfaces + e * Nfq * Nfaces * NFIELDS] = U.U2;
      Qvf[i + 2 * Nfq * Nfaces + e * Nfq * Nfaces * NFIELDS] = U.U3;
      Qvf[i + 3 * Nfq * Nfaces + e * Nfq * Nfaces * NFIELDS] = U.U4;
#if VDIM == 3
      Qvf[i + 4 * Nfq * Nfaces + e * Nfq * Nfaces * NFIELDS] = U.U5;
#endif
    }
  }

#if 1
  printf("Computing initial entropy\n");
  dfloat_t S0 = 0.0;
  for (uintloc_t e = 0; e < K; ++e)
  {
    for (int i = 0; i < Nq; ++i)
    {

      dfloat_t wJq = (app->hops->wq[i]) * (app->hops->Jq[i + e * Nq]);
      dfloat_t rho  = Q[i + 0 * Nq + e * Nq * NFIELDS];
      dfloat_t rhou = Q[i + 1 * Nq + e * Nq * NFIELDS];
      dfloat_t rhov = Q[i + 2 * Nq + e * Nq * NFIELDS];
      dfloat_t rhow = Q[i + 3 * Nq + e * Nq * NFIELDS];
      dfloat_t E    = Q[i + 4 * Nq + e * Nq * NFIELDS];

      dfloat_t gamma = app->prefs->physical_gamma;
      dfloat_t rhoe = E - .5f * (rhou * rhou + rhov * rhov + rhow * rhow) / rho;
      dfloat_t s = LOGDF((gamma - 1.0) * rhoe / POWDF(rho, gamma));
      S0 += wJq*(-rho*s);
    }
  }
#endif


  // copy mem to device
  printf("Copying to device\n");
#ifdef TESTING
  occaCopyPtrToMem(app->rhoLog, rhoLog, K*Nq*Nfq*Nfaces*app->prefs->kernel_KblkV*sizeof(dfloat_t), occaNoOffset);
  occaCopyPtrToMem(app->betaLog, betaLog, K*Nq*Nfq*Nfaces*app->prefs->kernel_KblkV*sizeof(dfloat_t), occaNoOffset);
  occaCopyPtrToMem(app->storage, storage, K*Nq*Nfq*Nfaces*app->prefs->kernel_KblkV*NFIELDS*sizeof(dfloat_t), occaNoOffset);
  occaCopyPtrToMem(app->store_pNq, store_pNq, K*Nq*Nfq*Nfaces*app->prefs->kernel_KblkV*sizeof(dfloat_t), occaNoOffset);
  occaCopyPtrToMem(app->store_pNfqNfaces, store_pNfqNfaces, K*Nfq*Nfaces*Nq*app->prefs->kernel_KblkV*sizeof(dfloat_t), occaNoOffset);
#endif
  occaCopyPtrToMem(app->Q, Q, Nq * NFIELDS * K * sizeof(dfloat_t),
                   occaNoOffset);
  occaCopyPtrToMem(app->rhsQ, Qvq, Nq * NFIELDS * K * sizeof(dfloat_t),
                   occaNoOffset);
  occaCopyPtrToMem(app->Qf, Qvf, Nfq * Nfaces * NFIELDS * K * sizeof(dfloat_t),
                   occaNoOffset);
  occaCopyPtrToMem(app->resQ, resQ, Nq * NFIELDS * K * sizeof(dfloat_t),
                   occaNoOffset);
  printf("Done copying to device\n");
  //  app_test(app); // testing
  //  return 0;

  // estimate time-step
  double hmin = get_hmin(app);
  double CFL = app->prefs->CFL;
  double FinalTime = app->prefs->FinalTime; // 1.0;//5.0; // 20*dt;

  double N = (double)app->prefs->mesh_N;
  double CN; // trace constant
#if VDIM == 2
  CN = (N + 1) * (N + 2) / 2;
#else
  CN = (N + 1) * (N + 3) / 3;
#endif
  double dt = CFL * hmin / CN;
  printf("hmin = %f, CFL = %f, dt = %f, Final Time = %f\n", hmin, CFL, dt, FinalTime);

  //test_rhs(app);  return 0;

#if 0

  app_test(app);
  const double rka = app->rk4a[0];
  const double rkb = app->rk4b[0];
  rk_step(app,rka,rkb,dt);
  app_test(app);

#else

  printf("Running...\n");
struct timespec start, end;
clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
  rk_run(app, dt, FinalTime);
clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
  printf("Done with simulation. \n");

printf("Elapsed Time: %f seconds\n", (float) ((1000000000.0*(end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec)/1000000000.0));

#endif

#if 1
  printf("Copying from mem\n");
  occaCopyMemToPtr(Q, app->Q, Nq * NFIELDS * K * sizeof(dfloat_t),
                   occaNoOffset);
#endif

  //#if 0
  //#endif

#if 1
  printf("Computing L2 error\n");
  dfloat_t err = 0.0;
  for (uintloc_t e = 0; e < K; ++e)
  {
    for (int i = 0; i < Nq; ++i)
    {

      coord X;
      X.x = app->hops->xyzq[i + 0 * Nq + e * Nq * 3];
      X.y = app->hops->xyzq[i + 1 * Nq + e * Nq * 3];
#if VDIM == 3
      X.z = app->hops->xyzq[i + 2 * Nq + e * Nq * 3];
#endif

      euler_fields Uex;
      euler_vortex(app, X, FinalTime, &Uex);

      dfloat_t wJq = (app->hops->wq[i]) * (app->hops->Jq[i + e * Nq]);
#if VDIM==2
      dfloat_t rho = Q[i + 0 * Nq + e * Nq * NFIELDS];
      dfloat_t rhou = Q[i + 1 * Nq + e * Nq * NFIELDS];
      dfloat_t rhov = Q[i + 2 * Nq + e * Nq * NFIELDS];
      dfloat_t E = Q[i + 3 * Nq + e * Nq * NFIELDS];

      dfloat_t err1 = (rho - Uex.U1);
      dfloat_t err2 = (rhou - Uex.U2);
      dfloat_t err3 = (rhov - Uex.U3);
      dfloat_t err4 = (E - Uex.U4);
      err += (err1 * err1 + err2 * err2 + err3 * err3 + err4 * err4) * wJq;
#else
      dfloat_t rho  = Q[i + 0 * Nq + e * Nq * NFIELDS];
      dfloat_t rhou = Q[i + 1 * Nq + e * Nq * NFIELDS];
      dfloat_t rhov = Q[i + 2 * Nq + e * Nq * NFIELDS];
      dfloat_t rhow = Q[i + 3 * Nq + e * Nq * NFIELDS];
      dfloat_t E    = Q[i + 4 * Nq + e * Nq * NFIELDS];

      dfloat_t err1 = (rho - Uex.U1);
      dfloat_t err2 = (rhou - Uex.U2);
      dfloat_t err3 = (rhov - Uex.U3);
      dfloat_t err4 = (rhow - Uex.U4);
      dfloat_t err5 = (E - Uex.U5);
      err += (err1 * err1 + err2 * err2 + err3 * err3 + err4 * err4 + err5 * err5) * wJq;
#endif
    }
  }
  printf("L2 err = %7.7g\n", sqrt(err));
#endif

#if VDIM==3
  printf("Computing norm of 3D solution \n");
  dfloat_t Unorm = 0.0;
  dfloat_t Sval = 0.0;
  for (uintloc_t e = 0; e < K; ++e)
  {
    for (int i = 0; i < Nq; ++i)
    {

#if 0
      coord X;
      X.x = app->hops->xyzq[i + 0 * Nq + e * Nq * 3];
      X.y = app->hops->xyzq[i + 1 * Nq + e * Nq * 3];
      X.z = app->hops->xyzq[i + 2 * Nq + e * Nq * 3];
#endif

      dfloat_t wJq = (app->hops->wq[i]) * (app->hops->Jq[i + e * Nq]);
      dfloat_t rho  = Q[i + 0 * Nq + e * Nq * NFIELDS];
      dfloat_t rhou = Q[i + 1 * Nq + e * Nq * NFIELDS];
      dfloat_t rhov = Q[i + 2 * Nq + e * Nq * NFIELDS];
      dfloat_t rhow = Q[i + 3 * Nq + e * Nq * NFIELDS];
      dfloat_t E    = Q[i + 4 * Nq + e * Nq * NFIELDS];

      dfloat_t gamma = app->prefs->physical_gamma;
      dfloat_t rhoe = E - .5f * (rhou * rhou + rhov * rhov + rhow * rhow) / rho;
      dfloat_t s = LOGDF((gamma - 1.0) * rhoe / POWDF(rho, gamma));

      Sval += wJq*(-rho*s);

      dfloat_t err1 = (rho );
      dfloat_t err2 = (rhou);
      dfloat_t err3 = (rhov);
      dfloat_t err4 = (rhow);
      dfloat_t err5 = (E );
      Unorm += (err1 * err1 + err2 * err2 + err3 * err3 + err4 * err4 + err5 * err5) * wJq;
    }
  }
  //printf("L2 norm of sol = %7.7g\n", sqrt(Unorm));
  printf("Initial entropy = %f, final entropy = %f, change in entropy = %7.7g\n", S0, Sval, fabs(Sval-S0));
#endif


// ============== print solution in matlab ===================
#if 0
  for (uintloc_t e = 0; e < K; ++e)
    {
      for (int i = 0; i < Nq; ++i)
	{
	  dfloat_t x = app->hops->xyzq[i + 0 * Nq + e * Nq * 3];
	  dfloat_t y = app->hops->xyzq[i + 1 * Nq + e * Nq * 3];
	  dfloat_t z = app->hops->xyzq[i + 2 * Nq + e * Nq * 3];
	  dfloat_t rho = Q[i + 0 * Nq + e * Nq * NFIELDS];

	  euler_fields Uex;
	  coord X; X.x = x; X.y = y; X.z = z;
	  euler_vortex(app, X, FinalTime, &Uex);

	  printf("x(%d,%d) = %f; y(%d,%d) = %f; z(%d,%d) = %f; rho(%d,%d) = %f; rhoex(%d,%d) = %f;\n",i+1,e+1,x,i+1,e+1,y,i+1,e+1,z,i+1,e+1,rho,i+1,e+1,Uex.U1);
	}
    }
#endif
  // ===========================================================

  //
  // cleanup
  //
  app_free(app);
  asd_free(app);
finalize:
  ASD_MPI_CHECK(MPI_Finalize());
  asd_gopt_free(options);

  return status;
}
// }}}
