#pragma once
#include "core/types.h"

namespace onn {

class PhaseDrift {
public:
    PhaseDrift(double sigma = 0.02);

    double get_phase_error(double dt);
    void reset();

    double sigma() const { return sigma_; }
    void set_sigma(double s) { sigma_ = s; }

private:
    double sigma_;
    double accumulated_ = 0.0;
};

} // namespace onn
