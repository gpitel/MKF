// Standalone benchmark of the OpenMagnetics MKF winding-loss field solve — the
// O(harmonics x turns^2 x mirrors) all-pairs Biot-Savart sum that dominates the
// simulation (MagneticField::calculate_magnetic_field_strength_field,
// MagneticField.cpp:394-592). This reproduces that exact hot loop and per-pair
// math (LAMMERANER round-wire model, MagneticField.cpp:923-957) on the real
// 1163-turn geometry, to measure the speedup and confirm identical output when
// the independent induced-point loop is parallelized with OpenMP.
//
// The induced-point loop is embarrassingly parallel: each induced point writes
// its own output element and the inner inducing-point sum runs entirely within
// one thread, so the floating-point result is bit-identical regardless of
// thread count (no cross-thread reduction).
//
// Build:  g++ -O3 -march=native -std=c++20 -fopenmp fieldbench.cpp -o fieldbench
// Run:    OMP_NUM_THREADS=1  ./fieldbench turns.txt     # baseline (serial)
//         OMP_NUM_THREADS=20 ./fieldbench turns.txt     # threaded

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <numbers>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Pt { double x, y, val; };

// LAMMERANER round-wire field of one inducing filament at one induced point
// (verbatim port of MagneticFieldStrengthLammeranerModel, MagneticField.cpp:931-956).
static inline void field_between(const Pt& ind /*inducing*/, const Pt& pt /*induced*/,
                                 double turnLength, double& Hx, double& Hy) {
    double distance = std::hypot(pt.y - ind.y, pt.x - ind.x);
    double angle    = std::atan2(pt.x - ind.x, pt.y - ind.y);
    double ex = std::cos(angle - std::numbers::pi / 2);
    double ey = std::sin(angle - std::numbers::pi / 2);
    double mod = -ind.val / 2 / std::numbers::pi / distance * turnLength / std::hypot(turnLength, distance);
    Hx = mod * ex;
    Hy = mod * ey;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "turns.txt";
    FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
    int n = 0;
    if (std::fscanf(f, "%d", &n) != 1) { std::fprintf(stderr, "bad header\n"); return 1; }
    std::vector<Pt> turns(n);
    for (int i = 0; i < n; ++i)
        if (std::fscanf(f, "%lf %lf %lf", &turns[i].x, &turns[i].y, &turns[i].val) != 3) { std::fprintf(stderr, "bad row %d\n", i); return 1; }
    std::fclose(f);

    // Window extent → mirror spacing (method of images, mirroring_dimension=1 → 3x3=9 images/turn).
    double minx = 1e30, maxx = -1e30, miny = 1e30, maxy = -1e30;
    for (auto& t : turns) { minx = std::min(minx, t.x); maxx = std::max(maxx, t.x); miny = std::min(miny, t.y); maxy = std::max(maxy, t.y); }
    double W = (maxx - minx) * 1.2 + 1e-3, H = (maxy - miny) * 1.2 + 1e-3;

    // Induced points = 1 per turn (round wire, CENTER model). Inducing points =
    // each turn + its 8 mirror images = 9x (CoilMesher.cpp:470-500).
    std::vector<Pt> induced = turns;
    std::vector<Pt> inducing;
    inducing.reserve(n * 9);
    for (auto& t : turns)
        for (int kx = -1; kx <= 1; ++kx)
            for (int ky = -1; ky <= 1; ++ky) {
                double sx = (kx == 0 ? 1.0 : -1.0), sy = (ky == 0 ? 1.0 : -1.0);
                inducing.push_back(Pt{ sx * t.x + kx * 2 * W, sy * t.y + ky * 2 * H, sx * sy * t.val });
            }

    const int harmonics = 4;                     // typical post-pruning harmonic count
    const double turnLength = 0.1;               // representative mean turn length (m)
    std::vector<double> outX((size_t)induced.size() * harmonics);
    std::vector<double> outY((size_t)induced.size() * harmonics);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    auto t0 = std::chrono::steady_clock::now();
    for (int h = 0; h < harmonics; ++h) {
        double hscale = 1.0 / (h + 1);           // harmonic amplitude falloff
        #pragma omp parallel for schedule(dynamic, 8)
        for (int ip = 0; ip < (int)induced.size(); ++ip) {
            double totX = 0, totY = 0;
            const Pt& p = induced[ip];
            for (size_t iq = 0; iq < inducing.size(); ++iq) {
                // skip self (same turn) — matches MagneticField.cpp:552-555
                if ((int)(iq / 9) == ip && iq % 9 == 4) continue;
                Pt ind = inducing[iq]; ind.val *= hscale;
                double Hx, Hy; field_between(ind, p, turnLength, Hx, Hy);
                totX += Hx; totY += Hy;
            }
            outX[(size_t)h * induced.size() + ip] = totX;
            outY[(size_t)h * induced.size() + ip] = totY;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    long double checksum = 0;
    for (size_t i = 0; i < outX.size(); ++i) checksum += std::fabs(outX[i]) + std::fabs(outY[i]);

    const char* env = std::getenv("OMP_NUM_THREADS");
    std::printf("turns=%d  inducing=%zu  induced=%zu  harmonics=%d  pair_evals=%.2fM  threads=%d(env=%s)  time_ms=%.1f  checksum=%.15g\n",
                n, inducing.size(), induced.size(), harmonics,
                (double)induced.size() * inducing.size() * harmonics / 1e6,
                threads, env ? env : "unset", ms, (double)checksum);
    return 0;
}
