import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time
import multiprocessing as mp
import pickle as pkl
from slowing_gaussian import enforce_thresholds_gaussian
import scipy.interpolate

def diff_keep_shape(x, t=None):
    shifted_left = np.concatenate((x[1:], [0]))
    shifted_right = np.concatenate(([0], x[:-1]))
    diff = (shifted_left - shifted_right) / 2.0

    # If time is given, scale the derivative
    if t is not None:
        t_shifted_left = np.concatenate((t[1:], [0]))
        t_shifted_right = np.concatenate(([0], t[:-1]))
        t_diff = (t_shifted_left - t_shifted_right) / 2.0

    diff[0] = x[1] - x[0]
    diff[-1] = x[-1] - x[-2]

    if t is not None:
        t_diff[0] = t[1] - t[0]
        t_diff[-1] = t[-1] - t[-2]
        diff /= t_diff

    return diff


def enforce_threshold(x, t, thresh, acc_limit):
    dt = (t[-1] - t[0]) / (t.shape[0] - 1)
    above_limit = x > thresh
    above_limit[0] = False
    above_limit[-1] = False

    first_above_limit = above_limit & np.logical_not(
        np.concatenate(([False], above_limit[:-1]))
    )
    last_above_limit = above_limit & np.logical_not(
        np.concatenate((above_limit[1:], [False]))
    )

    first_above_limit = np.where(first_above_limit)[0]
    last_above_limit = np.where(last_above_limit)[0]
    limit_ranges = np.stack((first_above_limit, last_above_limit), axis=1)

    # List that holds reconstruction info for each limit exceeding range
    reconstruction_info = [
        {"first_above": limit_ranges[i][0], "last_above": limit_ranges[i][1]}
        for i in range(limit_ranges.shape[0])
    ]

    # Compute the first derivative
    first_diff = diff_keep_shape(x, t)

    # For each limit exceeding range, find additional reconstruction info
    for i_segment in range(len(reconstruction_info)):
        # Find interesting point to the left
        i_first = reconstruction_info[i_segment]["first_above"]
        i_left = i_first - 1
        while i_left >= 0:
            # Check if the signal is below the threshold
            if x[i_left] < thresh:
                # Compute the acceleration required to stop below the threshold
                a_cand = (first_diff[i_left] ** 2) / (2 * (thresh - x[i_left]))

                # If the acceleration is below the limit, we have found the point
                if a_cand < acc_limit:
                    reconstruction_info[i_segment]["left"] = {
                        "index": i_left,
                        "x": x[i_left],
                        "v": first_diff[i_left],
                        "a": a_cand,
                        "dt": np.abs(first_diff[i_left]) / a_cand,
                    }
                    break

            i_left -= 1

        # Find interesting point to the right
        i_last = reconstruction_info[i_segment]["last_above"]
        i_right = i_last + 1
        while i_right < x.shape[0] - 1:
            # Check if the signal is below the threshold
            if x[i_right] < thresh:
                # Compute the acceleration required to stop below the threshold
                a_cand = (first_diff[i_right] ** 2) / (2 * (thresh - x[i_right]))

                # If the acceleration is below the limit, we have found the point
                if a_cand < acc_limit:
                    reconstruction_info[i_segment]["right"] = {
                        "index": i_right,
                        "x": x[i_right],
                        "v": first_diff[i_right],
                        "a": a_cand,
                        "dt": np.abs(first_diff[i_right]) / a_cand,
                    }
                    break

            i_right += 1

        # Reconstruct the intermediate part

        # Time between start of reconstruction and maximum
        dt_l = reconstruction_info[i_segment]["left"]["dt"]

        # Time between maximum and end of reconstruction
        dt_r = reconstruction_info[i_segment]["right"]["dt"]

        # Get number of samples
        num_samples = int(np.ceil((dt_l + dt_r) / dt) + 1)
        t_reconstructed = np.linspace(0, dt_l + dt_r, num_samples)

        # Sample both sides
        reconstructed_left = (
            reconstruction_info[i_segment]["left"]["x"]
            + reconstruction_info[i_segment]["left"]["v"] * t_reconstructed
            - 0.5 * reconstruction_info[i_segment]["left"]["a"] * t_reconstructed**2
        )
        t_reconstructed_with_offset = t_reconstructed - dt_l - dt_r
        reconstructed_right = (
            reconstruction_info[i_segment]["right"]["x"]
            + reconstruction_info[i_segment]["right"]["v"] * t_reconstructed_with_offset
            - 0.5
            * reconstruction_info[i_segment]["right"]["a"]
            * t_reconstructed_with_offset**2
        )
        reconstructed_both = (reconstructed_left * (t_reconstructed <= dt_l)) + (
            reconstructed_right * (t_reconstructed > dt_l)
        )
        t_reconstructed += t[reconstruction_info[i_segment]["left"]["index"]]
        reconstruction_info[i_segment]["t_reconstructed"] = t_reconstructed
        reconstruction_info[i_segment]["x_reconstructed"] = reconstructed_both

    # Stitch together the reconstructed parts
    t_list = []
    x_list = []
    for i_segment in range(len(reconstruction_info)):
        # Add the part in between reconstructions
        index_start = (
            0
            if i_segment == 0
            else reconstruction_info[i_segment - 1]["right"]["index"]
        )
        index_end = reconstruction_info[i_segment]["left"]["index"] + 1

        # Set the first sample to be at time previous + offset
        time_prev = 0.0 if i_segment == 0 else t_list[-1][-1]
        time_repairer = -t[index_start] + time_prev

        t_list.append(t[index_start:index_end] + time_repairer)
        x_list.append(x[index_start:index_end])

        # Add the reconstructed part (first sample at t_list[-1][-1]])
        time_repairer = (
            t_list[-1][-1] - reconstruction_info[i_segment]["t_reconstructed"][0]
        )

        t_with_firstlast = (
            reconstruction_info[i_segment]["t_reconstructed"] + time_repairer
        )
        x_with_firstlast = reconstruction_info[i_segment]["x_reconstructed"]

        t_list.append(t_with_firstlast)  # [1:-1])
        x_list.append(x_with_firstlast)  # [1:-1])

    # Add the last part
    index_start = reconstruction_info[-1]["right"]["index"]
    time_repairer = -t[index_start] + t_list[-1][-1]
    t_list.append(t[index_start:] + time_repairer)
    x_list.append(x[index_start:])

    # Remove overlaps
    for i_segment in range(len(x_list)):
        if i_segment % 2 == 1:
            t_list[i_segment] = t_list[i_segment][1:-1]
            x_list[i_segment] = x_list[i_segment][1:-1]


    # Concatenate and resample
    t = np.concatenate(t_list)
    x = np.concatenate(x_list)
    num_samples = int(np.ceil((t[-1] - t[0]) / dt))
    t_resampled = np.linspace(t[0], t[-1], num_samples)
    interp_fun = scipy.interpolate.interp1d(t, x)
    x_resampled = interp_fun(t_resampled)
    # x_resampled = np.interp(t_resampled, t, x)

    return x_resampled, t_resampled


def enforce_threshold_symmetric(x, t, thresh, acc_limit=None):
    # If no acceleration limit is given, use the maximum acceleration
    first_diff = diff_keep_shape(x, t)
    second_diff = diff_keep_shape(first_diff, t)
    if acc_limit is None:
        acc_limit = np.max(np.abs(second_diff))

    x, t = enforce_threshold(x, t, thresh, acc_limit)
    x, t = enforce_threshold(-x, t, thresh, acc_limit)
    return -x, t


def find_m(L, N, x_max, a_max, v_max, sigma=1.0, verbose=False):
    m_min = None
    m_max = None
    m_search = 100

    # Binary search for max m that does not violate the constraints
    while None in [m_min, m_max] or m_min < m_max - 1:
        # Get point of interest
        if None in [m_min, m_max]:
            m = m_search
        else:
            # Pick the middle point
            m = (m_min + m_max) // 2

        if verbose:
            print(
                f"Min: {str(m_min):5},  Mid: {str(m):5}, Max: {str(m_max):5}", end="\r"
            )

        # Compute the times
        t = np.linspace(-L / 2, L / 2, N)

        # # Generate the random coefficients
        a_rand = np.random.normal(0, 1 / (2.0 * m + 1), m)  # + 1)
        b_rand = np.random.normal(0, 1 / (2.0 * m + 1), m)  # + 1)

        vectorized = False
        if vectorized:
            a_rand = np.random.normal(0, 1 / (2.0 * m + 1), m)
            b_rand = np.random.normal(0, 1 / (2.0 * m + 1), m)
            x = np.sum(
                np.sqrt(2.0)
                * a_rand[:, None]
                * np.cos(2.0 * np.pi * np.arange(1, m + 1)[:, None] * t[None, :] / L)
                + np.sqrt(2.0)
                * b_rand[:, None]
                * np.sin(2.0 * np.pi * np.arange(1, m + 1)[:, None] * t[None, :] / L),
                axis=0,
            )
        else:
            a_rand = np.random.normal(0, 1 / (2.0 * m + 1), m + 1)
            b_rand = np.random.normal(0, 1 / (2.0 * m + 1), m + 1)
            x = np.zeros_like(t)
            for i in range(1, m + 1):
                x += np.sqrt(2) * a_rand[i] * np.cos(2.0 * np.pi * i * t / L) + np.sqrt(
                    2
                ) * b_rand[i] * np.sin(2.0 * np.pi * i * t / L)

        # Compute the standard deviations (x is centered around 0 with std 1)
        rescaling_factor = x_max / np.std(x)
        x = x * rescaling_factor

        v = diff_keep_shape(x, t=t)
        a = diff_keep_shape(v, t=t)

        x_std = np.std(x)  # = 1
        v_std = np.std(v)
        a_std = np.std(a)

        # Check if the constraints are violated
        is_violated = sigma * v_std > v_max or sigma * a_std > a_max
        if None in [m_min, m_max]:
            if m_min is None and m_max is None:
                if is_violated:
                    m_max = m
                    m_search = m // 2
                else:
                    m_min = m
                    m_search = m * 2
            elif m_min is None:
                if is_violated:
                    m_search = m // 2
                else:
                    m_min = m
            elif m_max is None:
                if is_violated:
                    m_max = m
                else:
                    m_search = m * 2
        elif is_violated:
            m_max = m
        else:
            m_min = m

    return m_min


def generate_single_joint(
    range_symm,
    max_acc,
    max_vel,
    T_target,
    warmup_time,
    num_samples,
    eps,
    dt_final=1e-2,
    verbose=False,
    id=-1,
):
    np.random.seed(id)

    if range_symm < eps:
        return None

    x_max = range_symm

    if verbose:
        print("Starting binary search for optimal target time")

    # Run binary search for the optimal target time
    T_range = [None, None]
    T_search_aux = None
    dT = 20.0

    final_run = False
    while True:
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Check if we are in the final run
        if not (
            T_range[0] is None or T_range[1] is None or T_range[1] - T_range[0] > dT
        ):
            print(f"[{time_str}] - {id} - Final run")
            T = (T_range[0] + T_range[1]) / 2.0
            N = int(np.round(T / dt_final)) + 1
            final_run = True
        else:
            # Nothing known yet
            if T_range[0] is None and T_range[1] is None:
                print(f"[{time_str}] - {id} - Initial run")
                T = T_target

            # Only the lower bound is known
            if T_range[0] is not None and T_range[1] is None:
                T = T_search_aux
                print(f"[{time_str}] - {id} - Increasing range: {T - T_range[0]:.1f}")

            # Only the upper bound is known
            if T_range[0] is None and T_range[1] is not None:
                T = T_search_aux
                print(f"[{time_str}] - {id} - Increasing range: {T_range[1] - T:.1f}")

            # Both bounds are known
            if T_range[0] is not None and T_range[1] is not None:
                T = (T_range[0] + T_range[1]) / 2.0
                print(
                    f"[{time_str}] - {id} - Decreasing range: {T_range[1] - T_range[0]:.1f}"
                )

            if verbose:
                print(
                    f"Min: {str(T_range[0]):6} Mid: {str(T):6}  Max: {str(T_range[1]):6}",
                    end="\r",
                )

            N = num_samples

            # Compute the optimal number of modes m
            m = find_m(T, num_samples, x_max, max_acc, max_vel, sigma=1, verbose=False)
            # print(f"M: {m}")

        # Generate the path
        L = T
        while True:
            t = np.linspace(-L / 2, L / 2, N)
            dt = t[1] - t[0]
            a_rand = np.random.normal(0, 1 / (2.0 * m + 1), m + 1)
            b_rand = np.random.normal(0, 1 / (2.0 * m + 1), m + 1)

            # Generate the path
            x = np.zeros_like(t)
            for i in range(1, m + 1):
                x += np.sqrt(2) * a_rand[i] * np.cos(2.0 * np.pi * i * t / L) + np.sqrt(
                    2
                ) * b_rand[i] * np.sin(2.0 * np.pi * i * t / L)

            # Rescale the path to the desired range
            rescaling_factor = x_max / np.std(x)
            x = x * rescaling_factor

            # Check if signal starts and ends within the range
            if np.abs(x[0]) < x_max and np.abs(x[-1]) < x_max:
                break

        # Apply the warump time
        warmup_steps = int(warmup_time / dt)
        warmup_01 = np.linspace(-np.pi / 2, np.pi / 2, warmup_steps)
        smooth_warmup = np.sin(warmup_01) / 2 + 0.5

        x[:warmup_steps] *= smooth_warmup
        x[-warmup_steps:] *= smooth_warmup[::-1]

        # Enforce velocity and acceleration limits
        x_constrained, t_constrained = enforce_thresholds_gaussian(
            x,
            t,
            v_max=max_vel,
            a_max=max_acc,
            agg="max",
            interp="quadratic",
        )

        # Enforce the position threshold
        try:
            x_constrained, t_constrained = enforce_threshold_symmetric(
                x_constrained, t_constrained, x_max, acc_limit=max_acc
            )
        except:
            time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"[{time_str}] - {id} - Failed to enforce threshold, retrying...")
            continue

        # Binary search for T
        is_too_long = t_constrained[-1] > T_target

        # Initial search, nothing known yet
        if T_range[0] is None and T_range[1] is None:
            if is_too_long:
                T_range[1] = T
            else:
                T_range[0] = T
            offset = -dT if is_too_long else dT
            T_search_aux = T + offset
            # print(f"Initial search, set interval to {T_search_aux}")
        elif T_range[0] is not None and T_range[1] is None:
            # Only the lower bound is known
            if is_too_long:
                # Super, we intialized the bounds for binary search
                T_range[1] = T
            else:
                # We need to increase the upper bound by doubling the distance to the lower bound
                distance = T - T_range[0]
                T_search_aux = T + distance
        elif T_range[0] is None and T_range[1] is not None:
            # Only the upper bound is known
            if not is_too_long:
                # Super, we intialized the bounds for binary search
                T_range[0] = T
            else:
                # We need to decrease the lower bound by doubling the distance to the upper bound
                distance = T_range[1] - T
                T_search_aux = T - distance
        elif T_range[0] is not None and T_range[1] is not None:
            # Both bounds are known
            if is_too_long:
                T_range[1] = T
            else:
                T_range[0] = T

        if final_run:
            break

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[{time_str}] - {id} - Done")

    # Append the results
    return [t_constrained, x_constrained]


def generate_paths(
    ranges,
    max_acc,
    max_vel,
    T,
    warmup_time,
    num_samples,
    eps=1e-6,
    dt_final=1e-2,
    visualize=False,
):
    dt = T / num_samples

    T_target = T
    ranges_center = np.mean(ranges, axis=1)
    ranges_symm = (ranges[:, 1] - ranges[:, 0]) / 2.0
    is_constant = ranges_symm < eps

    # Run pipeline for each joint
    parallel = True
    if parallel:
        args = [
            (
                ranges_symm[i],
                max_acc,
                max_vel,
                T,
                warmup_time,
                num_samples,
                eps,
                1e-2,
                i == 0,
                i,
            )
            for i in range(ranges.shape[0])
        ]
        with mp.Pool(len(args)) as pool:
            results = pool.starmap(generate_single_joint, args)
    else:
        results = [
            generate_single_joint(
                ranges_symm[i],
                max_acc,
                max_vel,
                T,
                warmup_time,
                num_samples,
                eps,
                1e-2,
                True,
            )
            for i in range(ranges.shape[0])
        ]

    # Resample according to the slowest joint
    t_max = -1
    i_joint_max = -1
    for i_joint in range(ranges.shape[0]):
        if results[i_joint] is None:
            continue

        # Set times to start at 0
        results[i_joint][0][:] -= results[i_joint][0][0]

        # Find the maximum time
        t_max = max(t_max, results[i_joint][0][-1])
        i_joint_max = i_joint

    # Rescale timescales
    for i_joint in range(ranges.shape[0]):
        if results[i_joint] is None:
            continue
        results[i_joint][0][:] /= results[i_joint][0][-1]
        results[i_joint][0][:] *= t_max

    # Re-centering
    for i_joint in range(ranges.shape[0]):
        if results[i_joint] is None:
            continue
        results[i_joint][1][:] += ranges_center[i_joint]

    # Do equidistant sampling
    times_new = np.linspace(0, t_max, int(t_max / dt_final) + 1)
    v_list = []
    a_list = []
    for i_joint in range(ranges.shape[0]):
        if results[i_joint] is None:
            results[i_joint] = [
                times_new,
                np.ones_like(times_new) * ranges_center[i_joint],
            ]
        else:
            interp_fun = scipy.interpolate.interp1d(
                results[i_joint][0], results[i_joint][1]
            )
            results[i_joint][1] = interp_fun(times_new)
            # results[i_joint][1] = np.interp(
            #     times_new, results[i_joint][0], results[i_joint][1]
            # )
            results[i_joint][0] = times_new
        v_list.append(diff_keep_shape(results[i_joint][1], t=results[i_joint][0]))
        a_list.append(diff_keep_shape(v_list[-1], t=results[i_joint][0]))

    # Formatting
    x_all = np.stack([res[1] for res in results], axis=-1)
    v_all = np.stack(v_list, axis=-1)
    a_all = np.stack(a_list, axis=-1)
    t_all = times_new

    res = {
        "angles": x_all,
        "velocities": v_all,
        "times": t_all,
        "time_max": t_max,
        "ranges": ranges,
    }

    # Save the results as npz
    os.makedirs("/output/smooth_random", exist_ok=True)
    np.savez("/output/smooth_random/paths.npz", **res)
    print("Saved trajectory to /output/smooth_random/paths.npz")

    # Plot everything 3 x N
    if visualize:
        plt.figure(figsize=(12, 8))

        for i_joint in range(ranges.shape[0]):
            # Plot position
            plt.subplot(3, ranges.shape[0], i_joint + 1)
            plt.plot(res["times"], res["angles"][:, i_joint])
            plt.axhline(ranges[i_joint, 0], color="k", linestyle="--")
            plt.axhline(ranges[i_joint, 1], color="k", linestyle="--")
            plt.title(f"$q_{i_joint}$")

            # Plot velocity histogram
            plt.subplot(3, ranges.shape[0], ranges.shape[0] + i_joint + 1)
            plt.hist(res["velocities"][:, i_joint], bins=50)
            plt.axvline(-max_vel, color="k", linestyle="--")
            plt.axvline(max_vel, color="k", linestyle="--")
            plt.title("$\\dot{q}_" + f"{i_joint}$")

            # Plot acceleration histogram
            plt.subplot(3, ranges.shape[0], 2 * ranges.shape[0] + i_joint + 1)
            plt.hist(a_all[:, i_joint], bins=50)
            plt.axvline(-max_acc, color="k", linestyle="--")
            plt.axvline(max_acc, color="k", linestyle="--")
            plt.title("$\\ddot{q}_" + f"{i_joint}$")

        plt.tight_layout()

        extensions = ["pdf", "png", "svg"]
        os.makedirs("/output/smooth_random", exist_ok=True)
        for ext in extensions:
            plt.savefig(f"/output/smooth_random/paths.{ext}")
            print(f"Saved plot to /output/smooth_random/paths.{ext}")

    return res


def main(duration_h=1):
    ranges = np.array(
        [
            [-np.pi, np.pi],
            [-1.04, 1.04],
            [-np.pi, np.pi],
            [-2.1, 2.1],
            [-np.pi, np.pi],
            [-2.44, 2.44],
            [-np.pi, np.pi],
        ]
    )

    max_vel = np.deg2rad(36.0)
    max_acc = 0.1 * max_vel

    T =  60 * 60.0 * duration_h
    num_samples_str = "1e5"
    num_samples = int(eval(num_samples_str))
    warmup_time = 30.0
    print(f"dt = {T/num_samples:.1e} s")

    time_start = time.perf_counter()
    res = generate_paths(
        ranges, max_acc, max_vel, T, warmup_time, num_samples, visualize=True
    )
    time_end = time.perf_counter()
    print(f"Found paths in {time_end - time_start:.3f} seconds                      ")



if __name__ == "__main__":
    main(duration_h=1)