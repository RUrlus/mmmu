/* bivariate_bounds.hpp -- Estimators of grid bounds using (marginal) Bivariate
 * Normal standard deviation. Copyright 2023 Ralph Urlus
 */
#pragma once

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <limits>

#include <mmu/common.hpp>
#include <mmu/confmat.hpp>

namespace mmu {
namespace core {

/**
 * \brief Compute mean and standard deviations for a metric accounting for edge
 cases.
 *
 * \note this function should only be used to determine the mean and standard
 deviation for computing the bounds as this method ensures sigma > 0.
 Whereas the true sigma can be equal to zero.
 *
 * \tparam x_idx index for first component of metric, e.g. `FPI` for precision
 * \tparam y_idx index for second component of metric, e.g. `TPI` for precision
 * \param[in] conf_mat confusion matrix
 * \param[out] mu the metric
 * \param[out] sigma the bounded standard deviation
 */
template <int x_idx, int y_idx>
inline void bvn_marginal_mu_sigma_for_bounds(
    const int_vt* __restrict const conf_mat, double& mu, double& sigma
) {
    double metric_for_sigma;
    const auto pair_sum
        = static_cast<double>(conf_mat[x_idx] + conf_mat[y_idx]);
    if (conf_mat[x_idx] == 0) {
        mu = 1.0;
        metric_for_sigma = (pair_sum - 1.0) / pair_sum;
        sigma = std::sqrt(
            (metric_for_sigma * (1.0 - metric_for_sigma)) / pair_sum
        );
    } else if (pair_sum > std::numeric_limits<double>::epsilon()) {
        mu = static_cast<double>(conf_mat[y_idx]) / pair_sum;
        sigma = std::sqrt(
            static_cast<double>(conf_mat[x_idx] * conf_mat[y_idx])
            / std::pow(pair_sum, 3.0)
        );
    } else {
        mu = 0.0;
        metric_for_sigma = 1.0 / pair_sum;
        sigma
            = std::sqrt((metric_for_sigma * (1 - metric_for_sigma)) / pair_sum);
    }
}  // bvn_marginal_mu_sigma_for_bounds

/**
 * \brief Compute scaled bounds based on marginal std. deviation.
 *
 * \tparam x_idx index for first component of metric, e.g. `FPI` for precision
 * \tparam y_idx index for second component of metric, e.g. `TPI` for precision
 * \param[in] conf_mat confusion matrix
 * \param[in] n_sigmas number of std.deviations to use as scaling factor
 * \param[in] epsilon value to constrain bounds to open set (0, 1)
 * \param[out] bounds the lower and upper bounds
 */
template <int x_idx, int y_idx>
inline void get_marginal_bounds(
    const int_vt* __restrict conf_mat,
    const double n_sigmas,
    const double epsilon,
    double* __restrict bounds
) {
    double mu;
    double sigma;
    bvn_marginal_mu_sigma_for_bounds<x_idx, y_idx>(conf_mat, mu, sigma);
    const double upper_epsilon = conf_mat[x_idx] == 0 ? 1.0 : 1.0 - epsilon;
    const double scaled_sigma = n_sigmas * sigma;
    bounds[0] = std::min(mu + scaled_sigma, upper_epsilon);
    bounds[1] = std::max(mu - scaled_sigma, epsilon);
}

/**
 * \brief Find the indexes that contain the scaled bounds based on marginal std.
 * deviation.
 *
 * \tparam x_idx index for first component of metric, e.g. `FPI` for precision
 * \tparam y_idx index for second component of metric, e.g. `TPI` for precision
 * \param[in] conf_mat confusion matrix
 * \param[in] vals values to compare the bounds to
 * \param[in] n_sigmas number of std.deviations to use as scaling factor
 * \param[in] epsilon value to constrain bounds to open set (0, 1)
 * \param[out] bound_idxs the index of the lower and upper bound in `vals`
 */
template <int x_idx, int y_idx>
inline void find_marginal_bounds(
    const int_vt* __restrict conf_mat,
    const double* __restrict vals,
    const int_vt n_bins,
    const double n_sigmas,
    const double epsilon,
    int_vt* __restrict bound_idxs
) {
    std::array<double, 2> bounds;
    // set min_prec, max_prec in bounds
    get_marginal_bounds<x_idx, y_idx>(
        conf_mat, n_sigmas, epsilon, bounds.data()
    );

    int_vt idx_min = 0;
    int_vt idx_max = n_bins;

    for (int_vt i = 0; i < n_bins; i++) {
        if (bounds[0] < vals[i]) {
            idx_min = i - 1;
            break;
        }
    }
    idx_min = idx_min > 0 ? idx_min : 0;
    bound_idxs[0] = idx_min;

    for (int_vt i = idx_min; i < n_bins; i++) {
        if (bounds[1] < vals[i]) {
            idx_max = i + 1;
            break;
        }
    }
    bound_idxs[1] = idx_max <= n_bins ? idx_max : n_bins;
}
}  // namespace core
}  // namespace mmu
