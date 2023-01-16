/* confmat.hpp -- Implementation of binary classification confusion matrix
 * Copyright 2023 Ralph Urlus
 */
#pragma once

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include <mmu/common.hpp>

namespace mmu {
namespace core {

/*                  pred
 *                0     1
 *  actual  0    TN    FP
 *          1    FN    TP
 *
 *  Flattened, implies C-contiguous, we have:
 */

/* True Negative Index of confusion matrix*/
constexpr int TNI = 0;
/* True Positive Index of confusion matrix*/
constexpr int FPI = 1;
/* False Negative Index of confusion matrix*/
constexpr int FNI = 2;
/* True Positive Index of confusion matrix*/
constexpr int TPI = 3;

/**
 * \brief Fill binary confusion matrix based on true labels y and estimated
 * labels yhat
 *
 * \note this function:
 * * does not handle nullptrs
 * * expects all memory to be contiguous
 * * expects conf_mat to point to zero'd memory
 *
 * \tparam T1 type of y, must be an integer type
 * \tparam T2 type of yhat, must be an integer type
 * \param[in] n_obs minimum length of y and yhat
 * \param[in] y the true labels
 * \param[in] yhat the estimated labels
 * \param[out] conf_mat confusion matrix to be filled
 */
template <typename T1, typename T2, isInt<T1> = true, isInt<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict yhat,
    int64_t* __restrict const conf_mat
) {
    for (int64_t i = 0; i < n_obs; i++) {
        conf_mat[static_cast<bool>(*y) * 2 + static_cast<bool>(*yhat)]++;
        yhat++;
        y++;
    }
}

/**
 * \brief Fill binary confusion matrix based on true labels y and estimated
 * classifier scores
 *
 * \note this function:
 * * does not handle nullptrs
 * * expects all memory to be contiguous
 * * expects conf_mat to point to zero'd memory
 *
 * \tparam T1 type of y, must be an integer type
 * \tparam T2 type of score and threshold, must be a floating type
 * \param[in] n_obs minimum length of y and yhat
 * \param[in] y the true labels
 * \param[in] score the classifier scores
 * \param[in] threshold the classifier/discrimination threshold
 * \param[out] conf_mat confusion matrix to be filled
 */
template <typename T1, typename T2, isInt<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    int64_t* __restrict const conf_mat
) {
    const double scaled_tol = GEQ_ATOL + GEQ_RTOL * threshold;
    for (int64_t i = 0; i < n_obs; i++) {
        conf_mat
            [static_cast<bool>(*y) * 2
             + geq_tol(*score, threshold, scaled_tol)]++;
        y++;
        score++;
    }
}

}  // namespace core
}  // namespace mmu
