"""Shared helpers for the analysis scripts: exact t critical values and LF-only text writing.

Every interval in this folder is a two-sided 95% Student-t interval over seeds, half-width = t(0.975, n-1) * sd / sqrt(n).
T975 holds the exact critical values for 1 to 11 degrees of freedom (n = 2 to 12 seeds). `check_constants()` recomputes them
with an independent standard-library implementation of the Student-t distribution (regularized incomplete beta function plus
bisection) and fails if any differs by more than 1e-9; reproduce.py runs it first.
"""
import math

T975 = {1: 12.7062047361747, 2: 4.30265272974946, 3: 3.18244630528371, 4: 2.77644510519779,
        5: 2.57058183563631, 6: 2.44691185114497, 7: 2.36462425159278, 8: 2.30600413520417,
        9: 2.26215716279820, 10: 2.22813885198627, 11: 2.20098516009164}


def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    p = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - p if t > 0 else p


def t_quantile(p, df):
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def check_constants():
    """Raise if any hard-coded critical value disagrees with the independent computation."""
    for df, value in T975.items():
        recomputed = t_quantile(0.975, df)
        if abs(recomputed - value) > 1e-9:
            raise AssertionError("T975[%d] = %r but t_quantile gives %r" % (df, value, recomputed))
    return True


def write_text_lf(path, text):
    """Write text with LF line endings and UTF-8 on every platform (so manifests that hash the bytes are portable)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
