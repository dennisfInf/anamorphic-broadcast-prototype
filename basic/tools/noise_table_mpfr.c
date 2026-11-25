/* Independent reference for the noise base table (src/noise.rs).
 *
 * For the rounded normal distribution round(N(0, s^2)) the table stores, for
 * k = 0, ..., B - 1, the 256-bit value
 *     T_k = floor(2^256 * P(|e| <= k) / P(|e| <= B))
 *         = floor(2^256 * erf((k + 1/2) / (s sqrt 2)) / erf((B + 1/2) / (s sqrt 2))).
 * This program evaluates that formula with MPFR at 1024-bit precision and
 * prints one entry per line as 64 hexadecimal digits, so that the output can be
 * compared with `noise_table --dump` byte for byte.
 *
 * The width is either a double SD, taken exactly, or var:V for s = sqrt(V)
 * with an integer variance V, computed at 1024 bits (the maintained profile
 * uses V = n (m + 1) = 34972416).
 *
 * Build and run (MPFR and GMP development files required):
 *     cc -O2 tools/noise_table_mpfr.c -lmpfr -lgmp -o noise_table_mpfr
 *     ./noise_table_mpfr var:34972416 111285 > mpfr.txt
 *     ./noise_table_mpfr 5913.748050094796 111285 > mpfr-double.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s SD|var:V BOUND\n", argv[0]);
        return 2;
    }
    const int exact = strncmp(argv[1], "var:", 4) == 0;
    const unsigned long variance = exact ? strtoul(argv[1] + 4, NULL, 10) : 0;
    const double sd = exact ? 0.0 : strtod(argv[1], NULL);
    const unsigned long bound = strtoul(argv[2], NULL, 10);
    const mpfr_prec_t prec = 1024;
    mpfr_t scale, x, total, c;
    mpfr_init2(scale, prec);
    mpfr_init2(x, prec);
    mpfr_init2(total, prec);
    mpfr_init2(c, prec);
    /* scale = s * sqrt(2): sqrt(2 V) for an exact variance, otherwise the
     * double sd (converted exactly) times sqrt(2). */
    if (exact) {
        mpfr_sqrt_ui(scale, 2 * variance, MPFR_RNDN);
    } else {
        mpfr_sqrt_ui(scale, 2, MPFR_RNDN);
        mpfr_set_d(x, sd, MPFR_RNDN);
        mpfr_mul(scale, scale, x, MPFR_RNDN);
    }
    mpfr_set_d(x, (double)bound + 0.5, MPFR_RNDN);
    mpfr_div(x, x, scale, MPFR_RNDN);
    mpfr_erf(total, x, MPFR_RNDN);
    char digits[80];
    for (unsigned long k = 0; k < bound; k++) {
        mpfr_set_d(x, (double)k + 0.5, MPFR_RNDN);
        mpfr_div(x, x, scale, MPFR_RNDN);
        mpfr_erf(c, x, MPFR_RNDN);
        mpfr_div(c, c, total, MPFR_RNDN);
        mpfr_mul_2ui(c, c, 256, MPFR_RNDN);
        mpfr_rint(c, c, MPFR_RNDD); /* floor */
        mpfr_exp_t exponent = 0;
        char *s = mpfr_get_str(NULL, &exponent, 16, 64, c, MPFR_RNDZ);
        /* s holds 64 significant digits of an integer with `exponent` digits. */
        if (exponent < 0 || exponent > 64) {
            fprintf(stderr, "entry %lu out of range\n", k);
            return 1;
        }
        memset(digits, '0', 64);
        digits[64] = '\0';
        memcpy(digits + 64 - exponent, s, (size_t)exponent);
        printf("%s\n", digits);
        mpfr_free_str(s);
    }
    mpfr_clear(scale);
    mpfr_clear(x);
    mpfr_clear(total);
    mpfr_clear(c);
    return 0;
}
