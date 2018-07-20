
if     ELEM_TYPE == 0 then meshname = "meshes/periodicSquare3.msh"
--elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCube0.msh"
--elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCube1.msh"
--elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCube2.msh"
--elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCube3.msh"
--elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCubeCoarse.msh"
--elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCubeTGCoarse.msh"
--elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCubeTG1.msh"
--elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCubeTG2.msh"
elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCubeTG3.msh"
--elseif ELEM_TYPE == 1 then meshname = "meshes/periodicCubeWithPeriodicSection.msh"
else
  print("Unknown element type")
end

app = {
  occa = {
    -- OpenMP
    -- info = "mode = Serial",
    -- flags = " -O2 -g -fno-common -fomit-frame-pointer \z
     --          -Wno-sign-conversion \z
     --          -Wcast-align -Wchar-subscripts -Wall -W \z
    --           -Wpointer-arith -Wwrite-strings -Wformat-security -pedantic \z
    --           -Wextra -Wno-unused-parameter -Wno-unknown-pragmas \z
     --          -Wno-unused-variable -Wno-int-in-bool-context"
     --          -- -Wconversion -fsanitize=address
    -- OpenCL
    -- info = string.format("mode = OpenCL , platformID = 0, deviceID = %d",
    --                      HOST_RANK),
    -- flags = " -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros \z
    --           -cl-unsafe-math-optimizations -cl-finite-math-only \z
    --           -cl-fast-relaxed-math"

    -- CUDA
     info = string.format("mode = CUDA , deviceID = 0"),
     flags = "--compiler-options -O3 --ftz=true --fmad=true"
    -- flags = "--compiler-options -O3 --ftz=true --prec-div=false \z
                                --prec-sqrt=false --use_fast_math \z
                                --fmad=true"
     --flags = "-g"
  },
  kernel = {
    KblkV = 1,
    KblkS = 1,
    KblkU = 1,
  },
  mesh = {
    filename = meshname,
    start_level = 1,
    N = 3,
    M = 6,
    sfc_partition = true
  },
  physical={
    FinalTime = 0.1,
    CFL = .25,
    tau = 1.0,
  },
  output = {
    datadir = "data",
  },
}
