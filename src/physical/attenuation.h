#pragma once
#include "core/types.h"
#include <Eigen/Dense>

namespace onn {

class Attenuation {
public:
    Attenuation(double db_per_cm = 0.3, double coupling_loss_db = 0.1);

    double apply(double power) const;
    double apply_db(double power_db) const;

    double total_loss_db(int num_couplers, int num_phase_shifters, double waveguide_cm) const;

    double db_per_cm() const { return db_per_cm_; }
    double coupling_loss_db() const { return coupling_loss_db_; }

private:
    double db_per_cm_;
    double coupling_loss_db_;
};

} // namespace onn
