#include "physical/shot_noise.h"

namespace onn {

ShotNoise::ShotNoise(double quantum_efficiency)
    : eta_(quantum_efficiency) {}

double ShotNoise::expected_photons(double power, double dt) const {
    double hbar_omega = PLANCK * SPEED_OF_LIGHT / TELECOM_WAVELENGTH;
    return eta_ * power * dt / hbar_omega;
}

int ShotNoise::sample(double power, double dt) const {
    double mean = expected_photons(power, dt);
    std::poisson_distribution<int> dist(static_cast<int>(std::max(0.0, mean)));
    return dist(RNG::instance().engine());
}

double ShotNoise::snr(double power, double dt) const {
    double mean = expected_photons(power, dt);
    return (mean > 0) ? std::sqrt(mean) : 0.0;
}

} // namespace onn
