/* pr_multn_loglike.hpp -- Implementation Multinomial uncertainty over
 * Precision-Recall using Wilk's theorem Copyright 2022 Ralph Urlus
 */
#include <mmu/multn_profile.hpp>

namespace mmu::core::pr {

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

void MultnPll::set_conf_mat(const int_vt* __restrict cm) {
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
    const auto [min_prec, max_prec, min_rec, max_rec]
        = get_bounds(conf_mat, n_sigmas, epsilon);
    bounds[0] = min_prec;
    bounds[1] = max_prec;
    bounds[2] = min_rec;
    bounds[3] = max_rec;
    // allocate memory for recall grid
    auto rec_grid = std::make_unique<double[]>(n_bins);
    // set recall grid
    linspace(min_rec, max_rec, n_bins, rec_grid.get());
    // determine step size for precision
    const double prec_start = bounds[0];
    const double prec_delta
        = (max_prec - min_prec) / static_cast<double>(n_bins - 1);

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

    MultnPll multn_pll(conf_mat);

    // obtain the indexes over which to loop
    const auto [idx_min_prec, idx_max_prec, idx_min_rec, idx_max_rec]
        = find_bound_idexes(
            conf_mat,
            prec_grid,
            rec_grid,
            n_prec_bins,
            n_rec_bins,
            n_sigmas,
            epsilon
        );

    for (int_vt i = idx_min_prec; i < idx_max_prec; i++) {
        multn_pll.set_precision(prec_grid[i]);
        int_vt odx = i * n_rec_bins;
        for (int_vt j = idx_min_rec; j < idx_max_rec; j++) {
            double score = multn_pll(rec_grid[j]);
            int_vt idx = odx + j;
            // log likelihoods and thus always positive
            if (score < scores[idx]) {
                scores[idx] = score;
            }
        }
    }
}  // multn_pll_grid_scores

/**
 * \brief Compute the minimum profile-likelihood scores over the confusion
 * matrices assuming a joint-Multinomial precision-recall distribution over a
 * predetermined grid.
 *
 * \note Assumes all arrays are C-contiguous
 *
 * \param[in] conf_mat the `n_conf_mats` confusion matrices
 * \param[in] prec_grid the precision values used to compute scores for
 * \param[in] rec_grid the recall values used to compute scores for
 * \param[in] n_conf_mats the number of confusion matrices to be evaluated
 * \param[in] n_prec_bins the maximum number of bins to evaluate for the y axis
 * \param[in] n_rec_bins the maximum number of bins to evaluate for the x axis
 * \param[in] n_sigmas the factor used to scale the grid, typical value is 6
 * \param[in] epsilon the distance from [0, 1] to limit grid on
 * \param[out] scores the minimum observed chi2 score per grid entry
 * profile-likelihood scores
 */
void multn_pll_grid_curve_scores(
    const int_vt* __restrict conf_mat,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int_vt n_conf_mats,
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

    auto mpl = MultnPll();

    auto bounds = GridBounds(
        n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid
    );

    for (int_vt k = 0; k < n_conf_mats; k++) {
        // update to new conf_mat
        mpl.set_conf_mat(conf_mat);
        bounds.find_bound_indexes(conf_mat);

        for (int_vt i = bounds.idx_min_prec; i < bounds.idx_max_prec; i++) {
            double prec = prec_grid[i];
            for (int_vt j = bounds.idx_min_rec; j < bounds.idx_max_rec; j++) {
                double score = mpl(prec, rec_grid[j]);
                int_vt idx = (i * n_rec_bins) + j;
                if (score < scores[idx]) {
                    scores[idx] = score;
                }
            }
        }
        // increment ptr
        conf_mat += 4;
    }
}  // multn_pll_grid_curve_scores

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
    const auto [min_prec, max_prec, min_rec, max_rec]
        = get_bounds(conf_mat, n_sigmas, epsilon);
    bounds[0] = min_prec;
    bounds[1] = max_prec;
    bounds[2] = min_rec;
    bounds[3] = max_rec;
    // allocate memory for recall grid
    auto rec_grid = std::make_unique<double[]>(n_bins);
    // set recall grid
    linspace(min_rec, max_rec, n_bins, rec_grid.get());
    // determine step size for precision
    const double prec_start = bounds[0];
    const double prec_delta
        = (max_prec - min_prec) / static_cast<double>(n_bins - 1);

    const MultnPll global_mpl(conf_mat);
#pragma omp parallel num_threads(n_threads) default(none) \
    shared(prec_start, prec_delta, n_bins, rec_grid, global_mpl, scores)
    {
        // each thread requires a copy as calls have side-effects
        MultnPll mpl(global_mpl);
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

/**
 * \brief Compute the minimum profile-likelihood scores over the confusion
 * matrices assuming a joint-Multinomial precision-recall distribution over a
 * predetermined grid.
 *
 * \note Assumes all arrays are C-contiguous
 *
 * \param[in] conf_mat the `n_conf_mats` confusion matrices
 * \param[in] prec_grid the precision values used to compute scores for
 * \param[in] rec_grid the recall values used to compute scores for
 * \param[in] n_conf_mats the number of confusion matrices to be evaluated
 * \param[in] n_prec_bins the maximum number of bins to evaluate for the y axis
 * \param[in] n_rec_bins the maximum number of bins to evaluate for the x axis
 * \param[in] n_sigmas the factor used to scale the grid, typical value is 6
 * \param[in] epsilon the distance from [0, 1] to limit grid on
 * \param[in] n_threads number of threads to use in the computation
 * \param[out] scores the minimum observed chi2 score per grid entry
 * profile-likelihood scores
 */
void multn_pll_grid_curve_scores_mt(
    const int_vt* __restrict conf_mat,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int_vt n_conf_mats,
    const int_vt n_prec_bins,
    const int_vt n_rec_bins,
    const double n_sigmas,
    const double epsilon,
    const int n_threads,
    double* __restrict scores
) {
    const int_vt n_elem = n_prec_bins * n_rec_bins;
    const int_vt t_elem = n_elem * n_threads;
    auto thread_scores = std::make_unique<double[]>(t_elem);

    // give scores a high enough initial value that the chi2 p-values will be
    // approximately zero
    std::fill(
        thread_scores.get(),
        thread_scores.get() + t_elem,
        MULTN_DEFAULT_CHI2_SCORE
    );
#pragma omp parallel num_threads(n_threads) default(none) shared( \
    n_elem,                                                       \
    n_prec_bins,                                                  \
    n_rec_bins,                                                   \
    n_conf_mats,                                                  \
    prec_grid,                                                    \
    rec_grid,                                                     \
    conf_mat,                                                     \
    n_sigmas,                                                     \
    thread_scores,                                                \
    epsilon                                                       \
)
    {
        double* thread_block
            = thread_scores.get() + (omp_get_thread_num() * n_elem);

        auto mpl = MultnPll();
        auto bounds = GridBounds(
            n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid
        );

#pragma omp for
        for (int_vt k = 0; k < n_conf_mats; k++) {
            const int_vt* lcm = conf_mat + (k * 4);
            // update to new conf_mat
            mpl.set_conf_mat(lcm);
            bounds.find_bound_indexes(lcm);

            for (int_vt i = bounds.idx_min_prec; i < bounds.idx_max_prec; i++) {
                double prec = prec_grid[i];
                int_vt odx = i * n_rec_bins;
                for (int_vt j = bounds.idx_min_rec; j < bounds.idx_max_rec;
                     j++) {
                    double score = mpl(prec, rec_grid[j]);
                    int_vt idx = odx + j;
                    if (score < thread_block[idx]) {
                        thread_block[idx] = score;
                    }
                }
            }
        }
    }  // omp parallel

    // collect the scores
    auto offsets = std::make_unique<int_vt[]>(n_threads);
    for (int_vt j = 0; j < n_threads; j++) {
        offsets[j] = j * n_elem;
    }

    for (int_vt i = 0; i < n_elem; i++) {
        double min_score = MULTN_DEFAULT_CHI2_SCORE;
        for (int_vt j = 0; j < n_threads; j++) {
            double tscore = thread_scores[i + offsets[j]];
            if (tscore < min_score) {
                min_score = tscore;
            }
        }
        scores[i] = min_score;
    }
}  // multn_pll_grid_curve_scores_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace mmu::core::pr
