//
// euler2d
//

// {{{ Headers
#include <errno.h>
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
#define UINTLOC_PRI PRIu32
#define UINTLOC_SCN SCNu32

typedef uint64_t uintglo_t;
#define occaUIntglo(x) occaULong((uintglo_t)(x))
#define occa_uintglo_name "unsigned long"
#define UINTGLO(x) ASD_APPEND(x, ull)
#define UINTGLO_MAX UINT64_MAX
#define UINTGLO_MAX_DIGITS INT64_MAX_DIGITS
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
#else
typedef float dfloat_t;
#define occaDfloat occaFloat
#define DFLOAT(x) ASD_APPEND(x, f)
#define DFLOAT_FMTe "24.16e"
#define DFLOAT_MAX FLT_MAX
#define DFLOAT_MPI MPI_FLOAT
#define DFLOAT_SQRT sqrtf
#endif
// }}}

// {{{ OCCA
/* OCCA modes */
#define SERIAL 0
#define OPENMP 1
#define OPENCL 2
#define CUDA 3
const char *const occa_modes[] = {"SERIAL", "OpenMP", "OpenCL", "CUDA", NULL};

/* OCCA language */
#define OKL_LANG (1 << 0)
#define OFL_LANG (1 << 1)
#define NATIVE_LANG (1 << 2)

static int get_occa_mode(const char *info)
{
  int mode = -1;
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
#define APP_NAME "euler2d"
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
  ASD_ROOT_INFO("compute precision = %d bytes", sizeof(double));
  ASD_ROOT_INFO("----------------------------------------------------------");
}
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
  char *mesh_filename;

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
      asd_lua_expr_string(L, "app.mesh.filename", "mesh.gmsh");
  prefs->mesh_N = (int)asd_lua_expr_integer(L, "app.mesh.N", 3);

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

// {{{ App
typedef struct app
{
  prefs_t *prefs;
  occaDevice device;
  occaStream copy;
  occaStream cmdx;
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
