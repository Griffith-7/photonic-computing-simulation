#include "physical/attenuation.h"

namespace onn {

Attenuation::Attenuation(double db_per_cm, double coupling_loss_db)
    : db_per_cm_(db_per_cm), coupling_loss_db_(coupling_loss_db) {}

double Attenuation::apply(double power) const {
    return power * std::pow(10.0, -db_per_cm_ / 10.0);
}

double Attenuation::apply_db(double power_db) const {
    return power_db - db_per_cm_;
}

double Attenuation::total_loss_db(int num_couplers, int num_phase_shifters,
                                   double waveguide_cm) const {
    return waveguide_cm * db_per_cm_ +
           num_couplers * coupling_loss_db_ +
           num_phase_shifters * 0.05;
}

} // namespace onn
