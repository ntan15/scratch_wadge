app = {
  occa = {
    -- OpenMP
    info = "mode = Serial",
    flags = " -O2 -g -fno-common -fomit-frame-pointer \z
              -Wno-sign-conversion \z
              -Wcast-align -Wchar-subscripts -Wall -W \z
              -Wpointer-arith -Wwrite-strings -Wformat-security -pedantic \z
              -Wextra -Wno-unused-parameter -Wno-unknown-pragmas \z
              -Wno-unused-variable"
              -- -Wconversion -fsanitize=address
    -- OpenCL
    -- info = string.format("mode = OpenCL , platformID = 0, deviceID = %d",
    --                      HOST_RANK),
    -- flags = " -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros \z
    --           -cl-unsafe-math-optimizations -cl-finite-math-only \z
    --           -cl-fast-relaxed-math"

    -- CUDA
    -- info = string.format("mode = CUDA , deviceID = %d", HOST_RANK),
    -- flags = "--compiler-options -O3 --ftz=true --prec-div=false \z
    --                            --prec-sqrt=false --use_fast_math \z
    --                            --fmad=true"
    -- flags = "-g",
  },
  mesh = {
    filename = "meshes/hole_tri.msh",
    start_level = 1,
    N = 5
  },
  output = {
    datadir = "data",
  },
}
