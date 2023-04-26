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

namespace mmu::core {

/*                  pred
 *                0     1
 *  actual  0    TN    FP
 *          1    FN    TP
 *
 *  Flattened, implies C-contiguous, we have: TN FP FN TP
 */

/**
 * \brief Fill confusion matrix based on true labels y and estimated
 * classifier scores
 *
 * \note this function:
 * * does not handle nullptrs
 * * expects all memory to be contiguous
 * * expects conf_mat to point to zero'd memory
 *
 * \tparam T1 type of y, must be an integer type
 * \tparam T2 type of score and threshold, must be a floating type
 * \tparam n_classes number of classes in the array
 * \param[in] n_obs minimum length of y and yhat
 * \param[in] y the true labels
 * \param[in] score the classifier scores
 * \param[in] threshold the classifier/discrimination threshold
 * \param[out] conf_mat confusion matrix to be filled
 */
template <
    typename T1,
    typename T2,
    const int n_classes,
    isInt<T1> = true,
    isFloat<T2> = true>
inline void confusion_matrix(
    const int_vt n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    int_vt* __restrict const conf_mat
) {
    const double scaled_tol = GEQ_ATOL + GEQ_RTOL * threshold;
    for (int_vt i = 0; i < n_obs; i++) {
        conf_mat
            [static_cast<bool>(*y) * n_classes
             + geq_tol(*score, threshold, scaled_tol)]++;
        y++;
        score++;
    }
}

/**
 * \brief Fill confusion matrix based on true labels y and estimated
 * classifier scores
 *
 * \note this function:
 * * does not handle nullptrs
 * * expects all memory to be contiguous
 * * expects conf_mat to point to zero'd memory
 *
 * \tparam T1 type of y, must be an integer type
 * \tparam T2 type of yhat, must be an integer type
 * \tparam n_classes number of classes in the array
 * \param[in] n_obs minimum length of y and yhat
 * \param[in] y the true labels
 * \param[in] yhat the estimated labels
 * \param[out] conf_mat confusion matrix to be filled
 */
template <
    typename T1,
    typename T2,
    const int n_classes,
    isInt<T1> = true,
    isInt<T2> = true>
inline void confusion_matrix(
    const int_vt n_obs,
    const T1* __restrict y,
    const T2* __restrict yhat,
    int_vt* __restrict const conf_mat
) {
    for (int_vt i = 0; i < n_obs; i++) {
        conf_mat
            [static_cast<bool>(*y) * n_classes + static_cast<bool>(*yhat)]++;
        yhat++;
        y++;
    }
}

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
    const int_vt n_obs,
    const T1* __restrict y,
    const T2* __restrict yhat,
    int_vt* __restrict const conf_mat
) {
    return confusion_matrix<T1, T2, 2>(n_obs, y, yhat, conf_mat);
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
    const int_vt n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    int_vt* __restrict const conf_mat
) {
    return confusion_matrix<T1, T2, 2>(n_obs, y, score, threshold, conf_mat);
}

}  // namespace mmu::core
