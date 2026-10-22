#include <math.h>
/* Exact modal recursion for a surface step response S(s) = D + a0*s + sum_i c_i (1 - exp(-k_i s)),
   driven by a flux q that is piecewise constant in the warped time s.
   y[k] = D*q[k] + z0 + sum_i z_i  evaluated at s_k (q[k] applied from s_k on), ds[k] = s_{k+1}-s_k. */
void modrec(int n, int m, const double *ds, const double *q, double D, double a0,
            const double *c, const double *kap, double *z, double *y)
{
    double z0 = 0.0;
    for (int i = 0; i < m; i++) z[i] = 0.0;
    for (int k = 0; k < n; k++) {
        double acc = z0;
        for (int i = 0; i < m; i++) acc += z[i];
        y[k] = D * q[k] + acc;
        double d = ds[k];
        z0 += a0 * q[k] * d;
        for (int i = 0; i < m; i++) {
            double e = exp(-kap[i] * d);
            z[i] = e * z[i] + c[i] * (1.0 - e) * q[k];
        }
    }
}
