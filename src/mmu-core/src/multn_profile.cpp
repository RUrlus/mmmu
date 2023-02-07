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


}  // namespace pr
}  // namespace core
}  // namespace mmu
