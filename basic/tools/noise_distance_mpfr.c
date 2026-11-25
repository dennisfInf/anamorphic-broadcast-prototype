/* Exact statistical distance between a noise base table of src/noise.rs and
 * the ideal rounded normal distribution round(N(0, s^2)).
 *
 * The table returns |e| = k with probability (T_k - T_{k-1}) / 2^256, where
 * T_{-1} = 0 and T_B = 2^256, and attaches a fair sign to k != 0. The ideal law
 * gives |e| = k with probability F(k) - F(k-1), F(k) = erf((k + 1/2)/(s sqrt 2)),
 * and the same fair sign. The total variation distance between the signed
 * laws therefore equals the one between the magnitudes,
 *     ( sum_{k <= B} |P_T(k) - P(k)| + 1 - F(B) ) / 2.
 * This program evaluates it with MPFR at 1024 bits. It also checks that every
 * entry is T_k = floor(2^256 F(k) / F(B)) and reports the largest deviation of
 * T_k from 2^256 F(k) / F(B) in units of 2^-256.
 *
 * WIDTH is a double SD (taken exactly) or var:V for s = sqrt(V). The dump comes
 * from `noise_table --dump PATH` (one 64-digit hexadecimal entry per line). The
 * optional third argument multiplies the distance by a number of samples, for
 * example m + 1 = 30358 errors per ciphertext.
 *
 * Build and run (MPFR and GMP development files required):
 *     cc -O2 tools/noise_distance_mpfr.c -lmpfr -lgmp -o noise_distance_mpfr
 *     ./noise_distance_mpfr var:34972416 table.txt 30358
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s SD|var:V DUMP [SAMPLES]\n", argv[0]);
        return 2;
    }
    const int exact = strncmp(argv[1], "var:", 4) == 0;
    const unsigned long variance = exact ? strtoul(argv[1] + 4, NULL, 10) : 0;
    const double sd = exact ? 0.0 : strtod(argv[1], NULL);
    const unsigned long samples = argc == 4 ? strtoul(argv[3], NULL, 10) : 0;
    FILE *in = fopen(argv[2], "r");
    if (!in) {
        perror("dump");
        return 1;
    }
    size_t capacity = 1024, bound = 0;
    char (*lines)[66] = malloc(capacity * sizeof *lines);
    char buffer[128];
    while (fgets(buffer, sizeof buffer, in)) {
        size_t len = strcspn(buffer, "\r\n");
        if (len == 0) continue;
        if (len != 64) {
            fprintf(stderr, "line %zu is not 64 hexadecimal digits\n", bound + 1);
            return 1;
        }
        if (bound == capacity) {
            capacity *= 2;
            lines = realloc(lines, capacity * sizeof *lines);
        }
        memcpy(lines[bound], buffer, 64);
        lines[bound][64] = '\0';
        bound++;
    }
    fclose(in);

    const mpfr_prec_t prec = 1024;
    mpfr_t scale, x, f, fb, prev_f, t, prev_t, two256, p, q, d, sum, dev, max_dev, fl, tail, out;
    mpfr_ptr all[] = {scale, x, f, fb, prev_f, t, prev_t, two256, p, q, d, sum, dev, max_dev, fl, tail, out};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) mpfr_init2(all[i], prec);
    if (exact) {
        mpfr_sqrt_ui(scale, 2 * variance, MPFR_RNDN);
    } else {
        mpfr_sqrt_ui(scale, 2, MPFR_RNDN);
        mpfr_set_d(x, sd, MPFR_RNDN);
        mpfr_mul(scale, scale, x, MPFR_RNDN);
    }
    mpfr_set_ui(two256, 1, MPFR_RNDN);
    mpfr_mul_2ui(two256, two256, 256, MPFR_RNDN);
    /* F(B) and the ideal tail mass 1 - F(B) = erfc((B + 1/2)/(s sqrt 2)). */
    mpfr_set_ui(x, 2 * bound + 1, MPFR_RNDN);
    mpfr_div_2ui(x, x, 1, MPFR_RNDN);
    mpfr_div(x, x, scale, MPFR_RNDN);
    mpfr_erf(fb, x, MPFR_RNDN);
    mpfr_erfc(tail, x, MPFR_RNDN);
    mpfr_set_ui(prev_f, 0, MPFR_RNDN);
    mpfr_set_ui(prev_t, 0, MPFR_RNDN);
    mpfr_set_ui(sum, 0, MPFR_RNDN);
    mpfr_set_ui(max_dev, 0, MPFR_RNDN);
    unsigned long mismatches = 0, decreasing = 0;
    for (size_t k = 0; k <= bound; k++) {
        if (k < bound) {
            mpfr_set_ui(x, 2 * k + 1, MPFR_RNDN);
            mpfr_div_2ui(x, x, 1, MPFR_RNDN);
            mpfr_div(x, x, scale, MPFR_RNDN);
            mpfr_erf(f, x, MPFR_RNDN);
            if (mpfr_set_str(t, lines[k], 16, MPFR_RNDN) != 0) {
                fprintf(stderr, "entry %zu is not hexadecimal\n", k);
                return 1;
            }
            /* exact entry 2^256 F(k) / F(B), its floor and the deviation of T_k. */
            mpfr_div(d, f, fb, MPFR_RNDN);
            mpfr_mul(d, d, two256, MPFR_RNDN);
            mpfr_rint(fl, d, MPFR_RNDD);
            if (mpfr_cmp(fl, t) != 0) mismatches++;
            mpfr_sub(dev, t, d, MPFR_RNDN);
            mpfr_abs(dev, dev, MPFR_RNDN);
            mpfr_max(max_dev, max_dev, dev, MPFR_RNDN);
        } else {
            mpfr_set(f, fb, MPFR_RNDN);
            mpfr_set(t, two256, MPFR_RNDN);
        }
        if (mpfr_cmp(t, prev_t) < 0) decreasing++;
        mpfr_sub(p, f, prev_f, MPFR_RNDN);       /* ideal P(|e| = k) */
        mpfr_sub(q, t, prev_t, MPFR_RNDN);
        mpfr_div(q, q, two256, MPFR_RNDN);       /* table P_T(|e| = k) */
        mpfr_sub(d, q, p, MPFR_RNDN);
        mpfr_abs(d, d, MPFR_RNDN);
        mpfr_add(sum, sum, d, MPFR_RNDN);
        mpfr_set(prev_f, f, MPFR_RNDN);
        mpfr_set(prev_t, t, MPFR_RNDN);
    }
    mpfr_add(sum, sum, tail, MPFR_RNDN);
    mpfr_div_2ui(sum, sum, 1, MPFR_RNDN);       /* total variation distance */
    printf("width %s, bound B = %zu\n", argv[1], bound);
    printf("entries equal to floor(2^256 F(k)/F(B)): %zu of %zu (mismatches %lu, decreasing steps %lu)\n",
           bound - mismatches, bound, mismatches, decreasing);
    printf("largest |T_k - 2^256 F(k)/F(B)|: %.6f units of 2^-256\n", mpfr_get_d(max_dev, MPFR_RNDN));
    mpfr_log2(out, tail, MPFR_RNDN);
    printf("ideal tail mass beyond B: 2^%.4f\n", mpfr_get_d(out, MPFR_RNDN));
    if (mpfr_sgn(sum) == 0) {
        printf("total variation distance per sample: 0\n");
    } else {
        mpfr_log2(out, sum, MPFR_RNDN);
        printf("total variation distance per sample: 2^%.4f (%.6e)\n",
               mpfr_get_d(out, MPFR_RNDN), mpfr_get_d(sum, MPFR_RNDN));
        if (samples > 0) {
            mpfr_mul_ui(sum, sum, samples, MPFR_RNDN);
            mpfr_log2(out, sum, MPFR_RNDN);
            printf("union bound for %lu samples: 2^%.4f\n", samples, mpfr_get_d(out, MPFR_RNDN));
        }
    }
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) mpfr_clear(all[i]);
    free(lines);
    return mismatches == 0 && decreasing == 0 ? 0 : 1;
}
