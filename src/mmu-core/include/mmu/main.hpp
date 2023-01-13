#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#
#include <cinttypes>  // for int64_t
#include <cmath>      // for ceil

namespace py = pybind11;

namespace mmu {

int add(int x, int y);
void bind_add(py::module& m);

py::array_t<int64_t> arange(int64_t start, int64_t end, int64_t step);
void bind_arange(py::module& m);

}  // namespace mmu
