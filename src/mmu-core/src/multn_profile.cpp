/* pr_multn_loglike.hpp -- Implementation Multinomial uncertainty over
 * Precision-Recall using Wilk's theorem Copyright 2022 Ralph Urlus
 */
#include <mmu/multn_profile.hpp>

namespace mmu {
namespace core {
namespace pr {

MultnPll::MultnPll() = default;
MultnPll::MultnPll(const int_vt* __restrict cm)
    : i_n3{cm[FPI] + cm[FNI] + cm[TPI]},
      n3{static_cast<double>(i_n3)},
      n{static_cast<double>(i_n3 + cm[TNI])},
      x_tn{static_cast<double>(cm[TNI])},
      x_fp{static_cast<double>(cm[FPI])},
      x_fn{static_cast<double>(cm[FNI])},
      x_tp{static_cast<double>(cm[TPI])},
      p_tn{x_tn / n},
      p_fp{x_fp / n},
      p_fn{x_fn / n},
      p_tp{x_tp / n},
      nll_h0{
          -2
          * (xlogy(x_tn, p_tn) + xlogy(x_fp, p_fp) + xlogy(x_fn, p_fn)
             + xlogy(x_tp, p_tp))} {}

void MultnPll::set_conf_mat(const int64_t* __restrict cm) {
    i_n3 = cm[1] + cm[2] + cm[3];
    n3 = static_cast<double>(i_n3);
    n = static_cast<double>(i_n3 + cm[0]);
    x_tn = static_cast<double>(cm[TNI]);
    x_fp = static_cast<double>(cm[FPI]);
    x_fn = static_cast<double>(cm[FNI]);
    x_tp = static_cast<double>(cm[TPI]);
    p_tn = x_tn / n;
    p_fp = x_fp / n;
    p_fn = x_fn / n;
    p_tp = x_tp / n;
    nll_h0 = -2
             * (xlogy(x_tn, p_tn) + xlogy(x_fp, p_fp) + xlogy(x_fn, p_fn)
                + xlogy(x_tp, p_tp));
}

double MultnPll::compute_pll() {
    p3_h1 = (n3 / n) * (1. / (1. + prec_ratio + rec_ratio));
    p2_h1 = rec_ratio * p3_h1;
    p1_h1 = prec_ratio * p3_h1;
    // guard against floating point noise resulting in negative
    // probabilities
    p0_h1 = std::max(1. - p1_h1 - p2_h1 - p3_h1, 0.0);

    nll_h1 = -2
             * (xlogy(x_tn, p0_h1) + xlogy(x_fp, p1_h1) + xlogy(x_fn, p2_h1)
                + xlogy(x_tp, p3_h1));
    return nll_h1 - nll_h0;
}

void MultnPll::set_precision(const double prec) {
    prec_ratio = (1.0 - prec) / prec;
}

double MultnPll::operator()(const double rec) {
    rec_ratio = (1.0 - rec) / rec;
    return compute_pll();
}

double MultnPll::operator()(const double prec, const double rec) {
    rec_ratio = (1.0 - rec) / rec;
    prec_ratio = (1.0 - prec) / prec;
    return compute_pll();
}

/**
 * \brief Compute the profile-likelihood scores assuming a joint-Multinomial
 * precision-recall distribution.
 *
 * \note Assumes all arrays are C-contiguous
 *
 * \param[in] conf_mat the confusion matrix
 * \param[in] n_bins number of bins per axis in the grid
 * \param[in] n_sigmas the factor used to scale the grid, typical value is 6
 * \param[in] epsilon the distance from [0, 1] to limit grid on
 * \param[out] bounds the boundaries of the grid with order [min, max] x
 [precision, recall]
 * \param[out] scores the chi2(2) distributed
 * profile-likelihood scores
 */
void multn_pll_scores(
    const int_vt* __restrict conf_mat,
    const int_vt n_bins,
    const double n_sigmas,
    const double epsilon,
    double* __restrict bounds,
    double* __restrict scores
) {
    // set min_prec, max_prec, min_rec, max_rec in bounds
    get_prec_rec_bounds(conf_mat, n_sigmas, epsilon, bounds);
    // allocate memory for recall grid
    auto rec_grid = std::unique_ptr<double[]>(new double[n_bins]);
    // set recall grid
    linspace(bounds[2], bounds[3], n_bins, rec_grid.get());
    // determine step size for precision
    const double prec_start = bounds[0];
    const double prec_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    MultnPll mpl(conf_mat);

    int_vt idx = 0;
    for (int_vt i = 0; i < n_bins; i++) {
        mpl.set_precision(prec_start + (static_cast<double>(i) * prec_delta));
        for (int_vt j = 0; j < n_bins; j++) {
            scores[idx] = mpl(rec_grid[j]);
            idx++;
        }
    }
}  // multn_pll_scores

/**
 * \brief Compute the profile-likelihood scores assuming a joint-Multinomial
 * precision-recall distribution over a predetermined grid.
 *
 * \note Assumes all arrays are C-contiguous
 *
 * \param[in] conf_mat the confusion matrix
 * \param[in] prec_grid the precision values used to compute scores for
 * \param[in] rec_grid the recall values used to compute scores for
 * \param[in] n_prec_bins the maximum number of bins to evaluate for the y axis
 * \param[in] n_rec_bins the maximum number of bins to evaluate for the x axis
 * \param[in] n_sigmas the factor used to scale the grid, typical value is 6
 * \param[in] epsilon the distance from [0, 1] to limit grid on
 * \param[out] scores the minimum observed chi2 score per grid entry
 * profile-likelihood scores
 */
void multn_pll_grid_scores(
    const int_vt* __restrict conf_mat,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int_vt n_prec_bins,
    const int_vt n_rec_bins,
    const double n_sigmas,
    const double epsilon,
    double* __restrict scores
) {
    // give scores a high enough initial value that the chi2 p-values will be
    // close to zero i.e. ppf(1-1e-14) --> 64.47398179869367
    std::fill(
        scores, scores + n_prec_bins * n_rec_bins, MULTN_DEFAULT_CHI2_SCORE
    );
    // -- memory allocation --
    std::array<int_vt, 4> bound_idxs;

    MultnPll multn_pll(conf_mat);
    // -- memory allocation --

    // obtain the indexes over which to loop
    // sets prec_idx_min, prec_idx_max, rec_idx_min, rec_idx_max
    find_prec_rec_bound_idxs(
        conf_mat,
        prec_grid,
        rec_grid,
        n_prec_bins,
        n_rec_bins,
        n_sigmas,
        epsilon,
        bound_idxs.data()
    );

    const int_vt prec_idx_min = bound_idxs[0];
    const int_vt prec_idx_max = bound_idxs[1];
    const int_vt rec_idx_min = bound_idxs[2];
    const int_vt rec_idx_max = bound_idxs[3];

    for (int_vt i = prec_idx_min; i < prec_idx_max; i++) {
        multn_pll.set_precision(prec_grid[i]);
        int_vt odx = i * n_rec_bins;
        for (int_vt j = rec_idx_min; j < rec_idx_max; j++) {
            double score = multn_pll(rec_grid[j]);
            int_vt idx = odx + j;
            // log likelihoods and thus always positive
            if (score < scores[idx]) {
                scores[idx] = score;
            }
        }
    }
}  // multn_pll_grid_scores
#ifdef MMU_HAS_OPENMP_SUPPORT

/**
 * \brief Compute the profile-likelihood scores assuming a joint-Multinomial
 * precision-recall distribution using multi-threading.
 *
 * \note Assumes all arrays are C-contiguous
 *
 * \param[in] conf_mat the confusion matrix
 * \param[in] n_bins number of bins per axis in the grid
 * \param[in] n_sigmas the factor used to scale the grid, typical value is 6
 * \param[in] epsilon the distance from [0, 1] to limit grid on
 * \param[in] n_threads number of threads to use in the computation
 * \param[out] bounds the boundaries of the grid with order [min, max] x
 [precision, recall]
 * \param[out] scores the chi2(2) distributed
 * profile-likelihood scores
 */
void multn_pll_scores_mt(
    const int_vt* __restrict conf_mat,
    const int_vt n_bins,
    const double n_sigmas,
    const double epsilon,
    const int n_threads,
    double* __restrict bounds,
    double* __restrict scores
) {
    // set min_prec, max_prec, min_rec, max_rec in bounds
    get_prec_rec_bounds(conf_mat, n_sigmas, epsilon, bounds);
    // allocate memory for recall grid
    auto rec_grid = std::unique_ptr<double[]>(new double[n_bins]);
    // set recall grid
    linspace(bounds[2], bounds[3], n_bins, rec_grid.get());
    // determine step size for precision
    const double prec_start = bounds[0];
    const double prec_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    const MultnPL global_mpl(conf_mat);
#pragma omp parallel num_threads(n_threads) default(none) \
    shared(prec_start, prec_delta, n_bins, rec_grid, global_mpl, scores)
    {
        // each thread requires a copy as calls have side-effects
        MultnPL mpl(global_mpl);
#pragma omp for
        for (int_vt i = 0; i < n_bins; i++) {
            int_vt idx = i * n_bins;
            mpl.set_precision(
                prec_start + (static_cast<double>(i) * prec_delta)
            );
            for (int_vt j = 0; j < n_bins; j++) {
                scores[idx + j] = mpl(rec_grid[j]);
            }
        }
    }  // omp parallel
}  // multn_pll_scores_mt

#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace pr
}  // namespace core
}  // namespace mmu
