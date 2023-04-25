#include <mmu/bivariate_bounds.hpp>

namespace mmu::core::pr {

GridBounds::GridBounds(
    const int n_prec_bins,
    const int n_rec_bins,
    const double n_sigmas,
    const double epsilon,
    const double* __restrict precs,
    const double* __restrict recs
)
    : n_prec_bins{n_prec_bins},
      n_rec_bins{n_rec_bins},
      n_sigmas{n_sigmas},
      epsilon{epsilon},
      precs{precs},
      recs{recs} {}

// /Brief Set min_val and max_val for the marginal
template <int x_axis, int y_axis>
void GridBounds::get_marginal_bounds(const int_vt* __restrict conf_mat) {
    std::tie(mu, sigma)
        = bvn_marginal_mu_sigma_for_bounds<x_axis, y_axis>(conf_mat);
    upper_epsilon = conf_mat[x_axis] == 0 ? 1.0 : 1.0 - epsilon;
    scaled_sigma = n_sigmas * sigma;
    min_val = std::min(mu + scaled_sigma, upper_epsilon);
    max_val = std::max(mu - scaled_sigma, epsilon);
}

template <int x_axis, int y_axis>
void GridBounds::find_marginal_bound_indexes(const int_vt* __restrict conf_mat
) {
    GridBounds::get_marginal_bounds<x_axis, y_axis>(conf_mat);

    if constexpr (x_axis == FPI) {
        vals = precs;
        n_bins = n_prec_bins;
    } else {
        vals = recs;
        n_bins = n_prec_bins;
    }

    idx_min = 0;
    idx_max = n_bins;

    for (int_vt i = 0; i < n_bins; i++) {
        if (min_val < vals[i]) {
            idx_min = i - 1;
            break;
        }
    }
    idx_min = idx_min > 0 ? idx_min : 0;

    for (int_vt i = idx_min; i < n_bins; i++) {
        if (max_val < vals[i]) {
            idx_max = i + 1;
            break;
        }
    }
    if constexpr (x_axis == FPI) {
        idx_min_prec = idx_min;
        idx_max_prec = idx_max <= n_bins ? idx_max : n_bins;
    } else {
        idx_min_rec = idx_min;
        idx_max_rec = idx_max <= n_bins ? idx_max : n_bins;
    }
}

/**
 * \brief Find the indexes of the bounds for precision and recall. The results
 * are set in `idx_min_*` and `idx_max_*`
 *
 * \param[in] conf_mat the confusion matrix
 */
void GridBounds::find_bound_indexes(const int_vt* __restrict conf_mat) {
    // Sets idx_min_prec, idx_max_prec
    GridBounds::find_marginal_bound_indexes<FPI, TPI>(conf_mat);
    // Sets idx_min_prec, idx_max_prec
    GridBounds::find_marginal_bound_indexes<FNI, TPI>(conf_mat);
}

}  // namespace mmu::core::pr
