# Weight Adjusted Discontinuous Galerkin with Energy Stability (WADGES)

## Building and running
This build assumes you have [GNU Make](https://www.gnu.org/software/make/).
To build the code you can run something like:
```sh
make -j
```
The code can be run in parallel simply with
```sh
mpirun -np 2 ./plumadg3d
```

## Third party code
### libocca
We use [libocca](http://libocca.org) for our concurrent compute abstraction.
An archive of the git repo is generated by runnning
```sh
./misc/scripts/get_occa
```
and included in the `vendor` directory.

### Lua
[Lua](http://lua.org) is used for the configuration file and is retrieved with
```sh
pushd vendor
curl -O https://www.lua.org/ftp/lua-5.3.4.tar.gz
popd
```
and included in the vendor directory.

## Code style
We use [`clang-format`](http://clang.llvm.org/docs/ClangFormat.html) to ensure
consistent formatting of the source code.  This can be enforced through git
hooks.  Execute the following shell command from the top level directory of the
project to install such a hook
```sh
ln -s $(pwd)/misc/git/hooks/pre-commit .git/hooks/pre-commit
```
