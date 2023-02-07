#include <mmu/bivariate_bounds.hpp>

namespace mmu {
namespace core {
namespace pr {

void get_prec_rec_bounds(
    const int_vt* __restrict conf_mat,
    const double n_sigmas,
    const double epsilon,
    double* __restrict bounds
) {
    // set min_prec, max_prec in bounds
    get_marginal_bounds<FPI, TPI>(conf_mat, n_sigmas, epsilon, bounds);
    // set min_rec, max_rec in bounds after precision
    get_marginal_bounds<FNI, TPI>(conf_mat, n_sigmas, epsilon, bounds + 2);
}

void find_prec_rec_bound_idxs(
    const int_vt* __restrict conf_mat,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int_vt n_prec_bins,
    const int_vt n_rec_bins,
    const double n_sigmas,
    const double epsilon,
    int_vt* __restrict bound_idxs
) {
    find_marginal_bounds<FPI, TPI>(
        conf_mat, prec_grid, n_prec_bins, n_sigmas, epsilon, bound_idxs
    );
    find_marginal_bounds<FNI, TPI>(
        conf_mat, rec_grid, n_rec_bins, n_sigmas, epsilon, bound_idxs + 2
    );
}

}  // namespace pr
}  // namespace core
}  // namespace mmu
