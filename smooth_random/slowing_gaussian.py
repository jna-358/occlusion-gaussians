import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.interpolate
import time


def get_acc(x, dt, alpha=1.0):
    dt_slowed = alpha * dt
    v = np.diff(x) / dt_slowed
    a = np.diff(v) / dt_slowed[:-1]
    return a


def get_vel(x, dt, alpha=1.0):
    dt_slowed = alpha * dt
    v = np.diff(x) / dt_slowed
    return v


def enforce_thresholds_gaussian(
    x, t, v_max=None, a_max=None, agg="max", interp="quadratic", verbose=False
):
    assert agg in ["max", "add"]
    assert interp in ["linear", "quadratic", "cubic"]

    dt = np.diff(t)

    v = get_vel(x, dt)
    a = get_acc(x, dt)

    if agg == "max":

        def agg_fun(x1, x2):
            return np.maximum(x1, x2)

    else:

        def agg_fun(x1, x2):
            return x1 + x2

    # Compute slowing factor for velocity
    alpha_pre_v = np.zeros_like(dt)
    if v_max is not None:
        # Get all peaks
        is_above = np.abs(v) > v_max
        is_above = np.concatenate([[False], is_above, [False]])
        is_diff = np.diff(is_above * 1.0)
        starts = np.where(is_diff > 0.5)[0]
        ends = np.where(is_diff < -0.5)[0]

        t_starts = t[starts]
        t_ends = t[ends]
        t_lengths_v = t_ends - t_starts
        t_centers = (t_starts + t_ends) / 2.0
        alpha_max_v = np.ones_like(t_centers)
        for i_ext in range(alpha_max_v.shape[0]):
            max_v = np.max(np.abs(v[starts[i_ext] : ends[i_ext] + 1]))
            alpha_max_v[i_ext] = max_v / v_max

        # Add gaussian peaks
        for i_ext in range(alpha_max_v.shape[0]):
            sigma = t_lengths_v[i_ext] / 1.0
            mu = t_centers[i_ext]
            alpha_pre_v_local = np.exp(-((t[:-1] - mu) ** 2) / (2 * sigma**2)) * (
                alpha_max_v[i_ext] - 1.0
            )

            alpha_pre_v = agg_fun(alpha_pre_v, alpha_pre_v_local)

    # Compute slowing factor for acceleration
    alpha_pre_a = np.zeros_like(dt)
    if a_max is not None:
        # Get all peaks
        is_above = np.abs(a) > a_max
        is_above = np.concatenate([[False], is_above, [False]])
        is_diff = np.diff(is_above * 1.0)
        starts = np.where(is_diff > 0.5)[0]
        ends = np.where(is_diff < -0.5)[0]
        t_starts = t[starts]
        t_ends = t[ends]
        t_lengths_a = t_ends - t_starts
        t_centers = (t_starts + t_ends) / 2.0
        alpha_max_a = np.ones_like(t_centers)

        for i_ext in range(alpha_max_a.shape[0]):
            max_a = np.max(np.abs(a[starts[i_ext] : ends[i_ext] + 1]))
            alpha_max_a[i_ext] = np.sqrt(max_a / a_max)

        # Add gaussian peaks
        for i_ext in range(alpha_max_a.shape[0]):
            sigma = t_lengths_a[i_ext] / 1.0
            mu = t_centers[i_ext]
            alpha_pre_a_local = np.exp(-((t[:-1] - mu) ** 2) / (2 * sigma**2)) * (
                alpha_max_a[i_ext] - 1.0
            )
            alpha_pre_a = agg_fun(alpha_pre_a, alpha_pre_a_local)

    # Combine both
    alpha_both = np.maximum(alpha_pre_v, alpha_pre_a)
    alpha = alpha_both + 1.0

    # Compute new times and resample equidistantly
    sampling_rate = (t[-1] - t[0]) / (t.shape[0] - 1)
    dt_slowed = alpha * dt
    t_slowed = np.ones_like(t) * t[0]
    t_slowed[1:] += np.cumsum(dt_slowed)
    t_slowed_equi = np.linspace(
        t_slowed[0], t_slowed[-1], int((t_slowed[-1] - t_slowed[0]) / sampling_rate)
    )
    t_start = time.perf_counter()

    interp_fun = scipy.interpolate.interp1d(t_slowed, x, kind=interp)
    x_slowed_equi = interp_fun(t_slowed_equi)

    t_done = time.perf_counter()
    if verbose:
        print(f"Interpolation took {t_done - t_start:.3f} s")

    total_slowdown = np.mean(alpha)
    perc_added = (total_slowdown - 1.0) * 100.0
    if verbose:
        print(f"Total slowdown: {total_slowdown:.2f}x ({perc_added:.2f}%)")

    return x_slowed_equi, t_slowed_equi


if __name__ == "__main__":
    np.random.seed(1)

    L = 35.0
    N = int(1e5)
    m = 20
    t = np.linspace(-L / 2, L / 2, N)

    # Generate the random coefficients
    a_rand = np.random.normal(0, 1 / (2.0 * m + 1), m + 1)
    b_rand = np.random.normal(0, 1 / (2.0 * m + 1), m + 1)

    # Generate the path
    x = np.zeros_like(t)
    for i in range(1, m + 1):
        x += np.sqrt(2) * a_rand[i] * np.cos(2.0 * np.pi * i * t / L) + np.sqrt(
            2
        ) * b_rand[i] * np.sin(2.0 * np.pi * i * t / L)

    v = np.diff(x) / np.diff(t)
    a = np.diff(v) / np.diff(t[:-1])
    a_max = 2.0 * np.std(a)
    v_max = 2.0 * np.std(v)

    x_slowed, t_slowed = enforce_thresholds_gaussian(
        x, t, v_max=v_max, a_max=a_max, verbose=True
    )

    v_slowed = get_vel(x_slowed, np.diff(t_slowed))
    a_slowed = get_acc(x_slowed, np.diff(t_slowed))

    plt.figure()
    plt.plot(t, x, label="x")
    plt.plot(t_slowed, x_slowed, label="x slowed")
    plt.title("x")
    plt.legend()

    plt.figure()
    plt.plot(t[:-1], v, label="v")
    plt.plot(t_slowed[:-1], v_slowed, label="v slowed")
    plt.axhline(v_max, color="k", linestyle="--")
    plt.axhline(-v_max, color="k", linestyle="--")
    plt.title("v")
    plt.legend()

    plt.figure()
    plt.plot(t[:-2], a, label="a")
    plt.plot(t_slowed[:-2], a_slowed, label="a slowed")
    plt.axhline(a_max, color="k", linestyle="--")
    plt.axhline(-a_max, color="k", linestyle="--")
    plt.title("a")
    plt.legend()

    plt.show()
