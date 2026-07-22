#pragma once
#include "core/types.h"

namespace onn {

class PhaseShifter {
public:
    PhaseShifter(double nominal_phase = 0.0, double error_std = 0.0);

    double nominal_phase() const { return nominal_phase_; }
    double actual_phase() const { return actual_phase_; }
    void set_nominal(double p);

    double voltage_to_phase(double voltage) const;
    double phase_to_voltage(double phase) const;

    void update(double dt, double drift_std);
    void apply_error(double error_std);

    void reset();

private:
    double nominal_phase_;
    double actual_phase_;
    double error_std_;
    double alpha_ = 1.0;
    double beta_ = 0.0;
};

} // namespace onn
