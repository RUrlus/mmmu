#include <mmu/bivariate_bounds.hpp>

namespace mmu::core::pr {

auto get_prec_rec_bounds(
    const int_vt* __restrict conf_mat,
    const double n_sigmas,
    const double epsilon
) {
    // set min_prec, max_prec
    auto [min_prec, max_prec]
        = get_marginal_bounds<FPI, TPI>(conf_mat, n_sigmas, epsilon);
    // set min_rec, max_rec
    auto [min_rec, max_rec]
        = get_marginal_bounds<FNI, TPI>(conf_mat, n_sigmas, epsilon);
    struct bounds {
        double min_prec, max_prec, min_rec, max_rec;
    };
    return bounds{min_prec, max_rec, min_rec, max_rec};
}

auto find_prec_rec_bound_idxs(
    const int_vt* __restrict conf_mat,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int_vt n_prec_bins,
    const int_vt n_rec_bins,
    const double n_sigmas,
    const double epsilon
) {
    auto [prec_lower_bound, prec_upper_bound] = find_marginal_bounds<FPI, TPI>(
        conf_mat, prec_grid, n_prec_bins, n_sigmas, epsilon
    );
    auto [rec_lower_bound, rec_upper_bound] = find_marginal_bounds<FNI, TPI>(
        conf_mat, rec_grid, n_rec_bins, n_sigmas, epsilon
    );

    struct indexes {
        int_vt prec_lb, prec_ub, rec_lb, rec_ub;
    };
    return indexes{
        prec_lower_bound, prec_upper_bound, rec_lower_bound, rec_upper_bound};
}

}  // namespace mmu::core::pr
