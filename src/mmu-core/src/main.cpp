#include <mmu/main.hpp>

namespace mmu {

int add(int x, int y) { return x + y; }

void bind_add(py::module& m) {
    m.def(
        "add",  // name of the function in python
        &add,   // reference to the C++ function
        // Docstring shown in Python
        R"pbdoc(Add two integers.  

        Parameters
        ----------
        x : int
            some integer
        y : int
            some other integer

        Returns
        -------
        int
            the sum of the two integers
        )pbdoc",
        py::arg("x"),  // argument names in Python
        py::arg("y")
    );
}

py::array_t<int64_t> arange(int64_t start, int64_t end, int64_t step) {
    int64_t size = static_cast<int64_t>(
        std::ceil((end - start) / static_cast<double>(step))
    );
    // create 1-d array of length `size`
    auto result = py::array_t<int64_t>(size);
    // get a mutable pointer-like object to the memory of the array
    auto result_ref = result.mutable_unchecked<1>();
    int64_t val = start;
    for (int64_t i = 0; i < size; i++) {
        result_ref[i] = val;
        val += step;
    }
    return result;
}

void bind_arange(py::module& m) {
    m.def(
        "arange",
        &arange,
        R"pbdoc(Create range of integers.

        Parameters
        ----------
        start : int
            lowest value in the range, inclusive
        end : int
            exclusive end of the range
        step : int
            size of the steps in the range

        Returns
        -------
        result : np.ndarray[int64]
            array of integers over [start, end)
        )pbdoc",
        py::arg("start"),
        py::arg("end"),
        py::arg("step") = 1  // argument with default value
    );
}

}  // namespace mmu
