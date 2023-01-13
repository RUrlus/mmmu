#include <pybind11/pybind11.h>
#include <mmu/main.hpp>

namespace py = pybind11;

namespace mmu {

PYBIND11_MODULE(_mmu_core, m) {
  bind_add(m);
  bind_arange(m);
}

}  // namespace mmu
