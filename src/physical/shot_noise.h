#pragma once
#include "core/types.h"
#include <Eigen/Dense>

namespace onn {

class ShotNoise {
public:
    ShotNoise(double quantum_efficiency = 0.9);

    int sample(double power, double dt) const;
    double expected_photons(double power, double dt) const;
    double snr(double power, double dt) const;

    double quantum_efficiency() const { return eta_; }
    void set_quantum_efficiency(double eta) { eta_ = eta; }

private:
    double eta_;
};

} // namespace onn
