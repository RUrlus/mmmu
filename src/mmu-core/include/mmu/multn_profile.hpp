/* pr_multn_loglike.hpp -- Implementation Multinomial uncertainty over
 * Precision-Recall using Wilk's theorem Copyright 2022 Ralph Urlus
 */
#pragma once

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include <mmu/bivariate_bounds.hpp>
#include <mmu/common.hpp>
#include <mmu/confmat.hpp>

namespace mmu {
namespace core {
namespace pr {

class MultnPll {
    int64_t i_n3;
    double n3;
    double n;
    double x_tn;
    double x_fp;
    double x_fn;
    double x_tp;
    double p_tn;
    double p_fp;
    double p_fn;
    double p_tp;
    double nll_h0;
    double nll_h1;
    double p0_h1;
    double p1_h1;
    double p2_h1;
    double p3_h1;
    double rec_ratio;
    double prec_ratio;

    double compute_pll();

   public:
    explicit MultnPll();
    explicit MultnPll(const int_vt* __restrict cm);
    void set_conf_mat(const int64_t* __restrict cm);
    void set_precision(const double prec);
    double operator()(const double rec);
    double operator()(const double prec, const double rec);
};

}  // namespace pr
}  // namespace core
}  // namespace mmu
