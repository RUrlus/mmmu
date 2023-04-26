/* common.hpp -- Utility functions and macros used in multiple headers.
 * Copyright 2023 Ralph Urlus
 */
#pragma once

#include <stdexcept>
#define UNUSED(x) (void)(x)

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) \
    || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OS_WIN
#endif

// handle error C2059: syntax error: ';'  on windows for this Macro
#ifndef OS_WIN
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#endif

// Fix for lack of ssize_t on Windows for CPython3.10
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127 \
)  // warning C4127: Conditional expression is constant
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <cmath>    // isnan, log
#include <cstring>  // for memset
#include <type_traits>

namespace mmu {

#if defined(MMU_USE_INT64)
using int_vt = int_fast64_t;
#else
using int_vt = int_fast32_t;
#endif

template <typename T>
using isInt = std::enable_if_t<std::is_integral<T>::value, bool>;

template <typename T>
using isFloat = std::enable_if_t<std::is_floating_point<T>::value, bool>;

namespace core {

/* True Negative Index of confusion matrix*/
constexpr int TNI = 0;
/* True Positive Index of confusion matrix*/
constexpr int FPI = 1;
/* False Negative Index of confusion matrix*/
constexpr int FNI = 2;
/* True Positive Index of confusion matrix*/
constexpr int TPI = 3;
/**
 * \Brief fill value for the chi2 scores, this values results p-values very
 * close to 1 chi2.ppf(1.-1e-14) --> 64.47398179869367
 */
constexpr double MULTN_DEFAULT_CHI2_SCORE = 65.0;

/** clamp value between lo and hi */
template <typename T>
inline const T& clamp(const T& v, const T& lo, const T& hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

/**
 * /Brief x times log of y.
 *
 * /Note returns 0.0 when x <= epsilon
 *
 * /tparam T type of x and y, T should be a floating point type
 * /param x value to multiply the log of y with
 * /param y value to take the log of
 * /return result x * log(y)
 */
template <typename T, isFloat<T> = true>
inline double xlogy(T x, T y) {
    if ((x <= std::numeric_limits<T>::epsilon()) && (!std::isnan(y))) {
        return 0.0;
    }
    return x * std::log(y);
}

/**
 * /Brief x times log of y.
 *
 * /Note returns 0.0 when x == 0
 *
 * /tparam T type of x and y, T should be an integer type
 * /param x value to multiply the log of y with
 * /param y value to take the log of
 * /return result x * log(y)
 */
template <typename T, isInt<T> = true>
inline double xlogy(T x, T y) {
    if (x == 0) {
        return 0.0;
    }
    return static_cast<double>(x) * std::log(static_cast<double>(y));
}

/**
 * /Brief Zero memory block.
 *
 * /tparam T type of elements ptr points to
 * /param ptr address of first element of array to be zero'd
 * /param nelem number of element in the array ptr points to
 */
template <typename T>
inline void zero_array(T* ptr, size_t n_elem) {
    memset(ptr, 0, n_elem * sizeof(T));
}

/**
 * /Brief Zero memory block.
 *
 * /Note This overload should be used when the number of elements
 * is known at compile time.
 *
 * /tparam T type of elements ptr points to
 * /tparam nelem number of element in the array ptr points to
 * /param ptr address of first element of array to be zero'd
 */
template <typename T, const size_t n_elem>
inline void zero_array(T* ptr) {
    memset(ptr, 0, n_elem * sizeof(T));
}

// default value for absolute tollerance for geq_tol
constexpr double GEQ_ATOL = 1e-8;
// default value for relative tollerance for geq_tol
constexpr double GEQ_RTOL = 1e-5;

/**
 * /Brief Check if a is greater or equal to b taking into account floating point
 * noise.
 *
 * /tparam T1 type of a, should be floating type
 * /tparam T2 type of b, should be floating type
 * /param a value to compare to b
 * /param b value to which a is compared to
 * /param scaled_tol tollerance, typically `absolute_tol + relative_tol * b`
 *
 * /return result true if a is greater or equal to b
 */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline bool geq_tol(const T1 a, const T2 b, const double scaled_tol) {
    const double delta = a - b;
    // the first condition checks if a is greater than b given the tollerance
    // the second condition checks if a and b are approximately equal
    return delta > scaled_tol || std::abs(delta) <= scaled_tol;
}

/**
 * /Brief Check if a is greater or equal to b taking into account floating point
 * noise.
 *
 * /Note that this function is assymmetric for the equality check as it uses
 * the scale of `b` to determine the tollerance.
 *
 * /tparam T1 type of a, should be floating type
 * /tparam T2 type of b, should be floating type
 * /param a value to compare to b
 * /param b value to which a is compared to
 * /param rtol relative tollerance scaled by b, typically 1e-5
 * /param atol absolute tollerance, typically 1e-8
 *
 * /return result true if a is greater or equal to b
 */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline bool geq_tol(
    const T1 a, const T2 b, const double rtol, const double atol
) {
    const double delta = a - b;
    const double scaled_tol = atol + rtol * b;
    // the first condition checks if a is greater than b given the tollerance
    // the second condition checks if a and b are approximately equal
    return delta > scaled_tol || std::abs(delta) <= scaled_tol;
}

/**
 * /Brief Create linearly spaced inclusive range.
 *
 * /param[in] start inclusive start of the range
 * /param[in] end inclusive end of the range
 * /param[in] steps number of steps in the range
 * /param[out] result address of array where range is to be stored
 */
inline void linspace(
    const double start, double const end, const int steps, double* result
) {
    if (steps == 0) {
        throw std::runtime_error("`steps` must be greater than zero.");
    } else if (steps == 1) {
        result[0] = static_cast<double>(start);
        return;
    }
    const int N = steps - 1;
    const double delta = (end - start) / static_cast<double>(N);
    result[0] = start;
    result[N] = end;
    for (int i = 1; i < N; ++i) {
        result[i] = start + (delta * i);
    }
    return;
}

}  // namespace core
}  // namespace mmu
