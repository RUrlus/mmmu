# mmu

This package was generated by `pyxt` to facilitate easy developments of C++ extesions (Pybind11) for Python.
All the boilerplate around CMake and compilation has been handled for you.

## The basics

The package name is `mmu`.
The compiled extension is named `_mmu_core`.

You can access the extension using `mmu.core` which is an alias for `mmu.lib._mmu_core` where the extension is stored.

## Installing the package

The package can be installed as normal using:

`pip install .`

## Code

All code in located in the `src` directory.
The directory structure follows the conventions for their respective languages.
Python code lives under `src/mmu`.
C++ code is placed `src/mmu-core/src` and `src/mmu-core/include/mmu`.

### Adding C++ functions

If you are adding a C++ function you have to follow these steps:

1. write implementation, e.g. `add` in `src/mmu-core/src/main.cpp`
2. write a binding function, e.g. `bind_add` in `src/mmu-core/src/main.cpp`
3. add the signatures to the corresponding header file, e.g. `add` and `bind_add` in `src/mmu-core/include/mmu/main.hpp`
4. add a call to the binding function, e.g. as `bind_add` in `src/mmu-core/src/bindings.cpp`
5. re-install package

See [Pybind11 docs](https://pybind11.readthedocs.io/en/stable/) for details on their package.
You can set the build-type in the `pyproject.toml`.

