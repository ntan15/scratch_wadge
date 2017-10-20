//
// euler
//

// {{{ Headers
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
#include "elements.h"
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

// {{{ Number Types
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

static dfloat_t strtodfloat_or_abort(const char *str)
{
  char *end;
  errno = 0;
  dfloat_t x = DFLOAT_STRTOD(str, &end);

  if (end == str)
    ASD_ABORT("%s: not a floating point number", str);
  else if ('\0' != *end)
    ASD_ABORT("%s: extra characters at end of input: %s", str, end);
  else if (ERANGE == errno)
    ASD_ABORT("%s out of range of type dfloat_t", str);

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
// }}}

// {{{ Solver Info
#define APP_NAME "euler"
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

  int mesh_N;
  int mesh_Nq;

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
  prefs->mesh_Nq = (int)asd_lua_expr_integer(L, "app.mesh.Nq", 3);

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
    0, 3, 2  // face 3
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
  dfloat_t *EToVX;  // element to vertex coordinates

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
  dfloat_t *VXglo = NULL;
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
        VXglo = asd_malloc_aligned(sizeof(dfloat_t) * VDIM * Nvglo);
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
        dfloat_t x = strtodfloat_or_abort(word);
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
          char *child = strtok(line, " ");
          char *mother = strtok(NULL, " ");

          ASD_TRACE("Periodic %s -> %s", child, mother);
          asd_dictionary_insert(&periodic_vertices, child, mother);
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
    ASD_TRACE("%5" UINTGLO_PRI " %" DFLOAT_FMTe " %" DFLOAT_FMTe, v,
              VXglo[VDIM * v + 0], VXglo[VDIM * v + 1]);

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
  mesh->EToVX = asd_malloc_aligned(sizeof(dfloat_t) * NVERTS * VDIM * E);

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
      char *mother = asd_dictionary_get_value(&periodic_vertices, child);

      if (mother)
      {
        const uintglo_t mv = strtouglo_or_abort(mother) - 1;
        ASD_TRACE("Setting child %ju to mother %ju", v, mv);
        v = mv;
      }

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
    uintloc_t offset;

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
    nm->EToVX = asd_malloc_aligned(sizeof(dfloat_t) * VDIM * NVERTS * nm->E);
    nm->EToVG = asd_malloc_aligned(sizeof(uintglo_t) * NVERTS * nm->E);

    for (int r = 0; r < size; ++r)
    {
      MPI_Irecv(nm->EToVX + VDIM * NVERTS * (nm->ER + nm->recv_starts[r]),
                VDIM * NVERTS * (nm->recv_starts[r + 1] - nm->recv_starts[r]),
                DFLOAT_MPI, r, 333, comm, recv_requests + r);

      MPI_Irecv(nm->EToVG + NVERTS * (nm->ER + nm->recv_starts[r]),
                NVERTS * (nm->recv_starts[r + 1] - nm->recv_starts[r]),
                UINTGLO_MPI, r, 333, comm, recv_requests + size + r);
    }

    dfloat_t *sendEToVX =
        asd_malloc_aligned(sizeof(dfloat_t) * VDIM * NVERTS * nm->ES);
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
                DFLOAT_MPI, r, 333, comm, send_requests + r);

      MPI_Isend(sendEToVG + NVERTS * nm->send_starts[r],
                NVERTS * (nm->send_starts[r + 1] - nm->send_starts[r]),
                UINTGLO_MPI, r, 333, comm, send_requests + size + r);
    }

    memcpy(nm->EToVX, om->EToVX, sizeof(dfloat_t) * VDIM * NVERTS * nm->ER);
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
        fprintf(file, " %" DFLOAT_FMTe,
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
  int i, j;
  for (i = 0; i < VDIM; ++i)
    H[i] = 0;

  for (i = 0; i < VDIM; ++i)
  {
    for (j = 0; j < UINTGLO_BITS; ++j)
    {
      const int k = i * UINTGLO_BITS + j;
      const uint64_t bit = (X[VDIM - 1 - (k % VDIM)] >> (k / VDIM)) & 1;
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
  dfloat_t *EToC = asd_malloc_aligned(sizeof(dfloat_t) * VDIM * om->E);
  dfloat_t cmaxloc[VDIM] = {-DFLOAT_MAX}, cminloc[VDIM] = {DFLOAT_MAX};
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
      printf(" %" DFLOAT_FMTe, EToC[e * VDIM + d]);
    printf("\n");
  }
#endif

  // {{{ compute Hilbert integer centroids
  dfloat_t cmaxglo[VDIM] = {-DFLOAT_MAX}, cminglo[VDIM] = {DFLOAT_MAX};
  // These calls could be joined
  ASD_MPI_CHECK(
      MPI_Allreduce(cmaxloc, cmaxglo, VDIM, DFLOAT_MPI, MPI_MAX, comm));
  ASD_MPI_CHECK(
      MPI_Allreduce(cminloc, cminglo, VDIM, DFLOAT_MPI, MPI_MIN, comm));

#if 0
  printf("min:\n");
  for (int d = 0; d < VDIM; ++d)
    printf(" %" DFLOAT_FMTe, cminglo[d]);
  printf("\n");

  printf("max:\n");
  for (int d = 0; d < VDIM; ++d)
    printf(" %" DFLOAT_FMTe, cmaxglo[d]);
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
    uintloc_t offset;

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
  dfloat_t *bufEToVX =
      asd_malloc_aligned(sizeof(dfloat_t) * NVERTS * VDIM * om->E);

  nm->E = part_E[rank];
  nm->EToVG = asd_malloc_aligned(sizeof(uintglo_t) * NVERTS * nm->E);
  nm->EToVX = asd_malloc_aligned(sizeof(dfloat_t) * NVERTS * VDIM * nm->E);

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
              DFLOAT_MPI, r, 333, comm, recv_requests + size + r);

  for (int r = 0; r < size; ++r)
    MPI_Isend(bufEToVG + NVERTS * part_send_starts[r],
              NVERTS * (part_send_starts[r + 1] - part_send_starts[r]),
              UINTGLO_MPI, r, 333, comm, send_requests + r);

  for (int r = 0; r < size; ++r)
    MPI_Isend(bufEToVX + NVERTS * VDIM * part_send_starts[r],
              NVERTS * VDIM * (part_send_starts[r + 1] - part_send_starts[r]),
              DFLOAT_MPI, r, 333, comm, send_requests + size + r);

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
  prefs_t *prefs;
  occaDevice device;
  occaStream copy;
  occaStream cmdx;

  host_mesh_t *hm;
} app_t;

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

  return app;
}

static void app_free(app_t *app)
{
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

  //
  // run
  //
  // foo(0);
#if ELEM_TYPE == 0 // triangle
  build_operators_C_2D(app->prefs->mesh_N, app->prefs->mesh_Nq);
#else
  build_operators_C_3D(app->prefs->mesh_N, app->prefs->mesh_Nq);
#endif

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
