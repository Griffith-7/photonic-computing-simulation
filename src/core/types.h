#pragma once
#include <complex>
#include <cmath>
#include <numbers>
#include <random>

namespace onn {

using Complex = std::complex<double>;
inline constexpr Complex IM{0.0, 1.0};
inline constexpr double PI = std::numbers::pi;
inline constexpr double TWO_PI = 2.0 * std::numbers::pi;
inline constexpr double SPEED_OF_LIGHT = 299792458.0;
inline constexpr double PLANCK = 6.62607015e-34;
inline constexpr double TELECOM_WAVELENGTH = 1550e-9;

inline double wavelength_to_frequency(double lambda) {
    return SPEED_OF_LIGHT / lambda;
}

inline double wavelength_to_energy(double lambda) {
    return PLANCK * SPEED_OF_LIGHT / lambda;
}

struct Wave {
    Complex amplitude{0.0, 0.0};
    double wavelength = TELECOM_WAVELENGTH;

    double intensity() const { return std::norm(amplitude); }
    double phase() const { return std::arg(amplitude); }
    double power_dbm() const {
        double p = intensity();
        return (p > 0) ? 10.0 * std::log10(p * 1000.0) : -1000.0;
    }

    Wave normalized() const {
        double mag = std::abs(amplitude);
        return (mag > 0) ? Wave{amplitude / mag, wavelength} : Wave{Complex{0}, wavelength};
    }

    Wave operator*(const Wave& o) const {
        return Wave{amplitude * o.amplitude, wavelength};
    }
    Wave operator+(const Wave& o) const {
        return Wave{amplitude + o.amplitude, wavelength};
    }
    Wave operator-(const Wave& o) const {
        return Wave{amplitude - o.amplitude, wavelength};
    }
    Wave operator*(double s) const {
        return Wave{amplitude * s, wavelength};
    }
};

class RNG {
public:
    static RNG& instance() {
        static RNG r;
        return r;
    }
    std::mt19937& engine() { return engine_; }
    void seed(unsigned s) { engine_.seed(s); }

private:
    RNG() : engine_(std::random_device{}()) {}
    std::mt19937 engine_;
};

inline double gaussian_noise(double std_dev) {
    std::normal_distribution<double> dist(0.0, std_dev);
    return dist(RNG::instance().engine());
}

inline double uniform_noise(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(RNG::instance().engine());
}

} // namespace onn
