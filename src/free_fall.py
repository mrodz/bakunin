import math
import numpy as np
import matplotlib.pyplot as plt


def instantaneous_velocity(data: list[float], observe_at_n_moments: int, *, derivative_delta_divisor: float = 1) -> (
        dict)[tuple[float, int, int], float]:
    result = {}

    for i in range(1, observe_at_n_moments + 1):
        moment = len(data) / (observe_at_n_moments + 1) * i
        upper_t = min(math.ceil(moment), len(data) - 1)
        lower_t = math.floor(moment)

        if upper_t == lower_t:
            if upper_t == len(data) - 1:
                continue

            upper_t += 1
            lower_t -= 1

        upper = data[upper_t]
        lower = data[lower_t]

        result[(moment, lower_t, upper_t)] = (upper - lower) / ((upper_t - lower_t) / derivative_delta_divisor)

    return result


def instantaneous_acceleration(data: list[float], observe_at_n_moments: int,
                               *, derivative_delta_divisor: float = 1) -> dict[float, float]:
    result = {}

    velocities = instantaneous_velocity(data, observe_at_n_moments + 1,
                                        derivative_delta_divisor=derivative_delta_divisor)

    for i in range(2, observe_at_n_moments + 1):
        moment = len(velocities) / (observe_at_n_moments + 1) * i
        upper_t = math.ceil(moment)
        lower_t = math.floor(moment)

        if upper_t == lower_t:
            upper_t += 1
            lower_t -= 1

        slope_edges = []
        t_delta = []

        for (instant, floored_index, ceiled_index), slope in velocities.items():
            print(f'{upper_t}/{lower_t} @ {floored_index}<{ceiled_index}: {slope}')
            if floored_index <= upper_t <= ceiled_index:
                slope_edges.append(slope)
                t_delta.append(instant)

            if floored_index <= lower_t <= ceiled_index:
                slope_edges.append(slope)
                t_delta.append(instant)

        if len(slope_edges) != 2:
            raise Exception

        upper, lower = slope_edges
        upper_t_delta, lower_t_delta = t_delta

        print(upper, lower, upper_t_delta, lower_t_delta)

        result[moment] = (upper - lower) / ((upper_t_delta - lower_t_delta) / derivative_delta_divisor)
        print("=", result[moment])

    return result


def displacement(t: float, v0: float, a: float):
    return v0 * t + (1 / 2 * a) * t ** 2


POINTS_OF_OBSERVATION = 8


def main():
    _our_data = np.array(
        [0, 0.002, 0.004, 0.006, 0.011, 0.017, 0.026, 0.036, 0.048, 0.061, 0.076, 0.094, 0.113, 0.134, 0.157, 0.182,
         0.209, 0.235, 0.265, 0.297, 0.330, 0.365, 0.403, 0.442, 0.484, 0.524, 0.574, 0.625, 0.668, 0.717, 0.767,
         0.823, 0.870, 0.925])

    _fake_data = [0, 0.001, 0.005, 0.012, 0.019, 0.026, 0.037, 0.051, 0.065, 0.079, 0.105,
                  0.135, 0.169, 0.195, 0.231,
                  0.274, 0.314, 0.352, 0.395, 0.453, 0.505, 0.546, 0.592, 0.625, 0.668, 0.695, 0.735, 0.774,
                  0.839, 0.880, 0.925, 0.985, 0.985, 0.985]

    _daniel_data = [
        0, 0, 0, .001, .002, .004, .007, .012, .018, .026, .036, .047, .061, .076, .095, .111, .132, .154, .177, .202,
        .230, .266, .290,
        .322, .357, .399, .434, .472, .514, .556, .606, .658, .704, .754, .805, .859, .911
    ]

    _ava_data = [
        0.0, .003, .006, .009, .014, .020, .029, .040, .051, .065, .081, .1, .119, .142, .166, .192, .220, .249, .281,
        .314, .349, .386, .426, .466, .508, .555, .607, .651, .705, .758, .810, .865, .915
    ]

    data = _our_data

    time = [i / 60 for i in range(len(data))]

    plt.plot(time, data)

    expected_x = []
    for t in time:
        expected_x.append(displacement(t, 0, 9.81))

    plt.plot(time, expected_x)
    plt.show()

    velocities = []
    expected_velocities = []

    for i in range(1, len(data)):
        l, r = data[i - 1:i + 1]
        slope = (r - l) / (1 / 60)
        velocities.append(slope)
        expected_velocities.append(9.81 * i / 60)
        print(f"x' = ({r:.3f} - {l:.3f}) / (1/60) =", slope)

    plt.plot(time[:-1], velocities)
    plt.plot(time[:-1], expected_velocities)
    plt.show()

    accelerations = []

    for i in range(1, len(velocities)):
        l, r = velocities[i - 1:i + 1]
        slope = (r - l) / (1 / 60)
        accelerations.append(slope)
        print(f"x'' = ({r:.3f} - {l:.3f}) / (1/60) =", slope)

    plt.plot(time[:-2], accelerations)
    plt.plot(time[:-2], [9.81 for _ in range(len(data) - 2)])
    plt.show()

    for i in range(len(data)):
        print(f"t={i}/60\t\t{-data[i]:.3f} | {displacement(i / 60, 0, -6.7):.3f}")

    for i in range(1, len(velocities) - 1):
        print(i, "->", (velocities[i + 1] - (2 * velocities[i]) + velocities[i - 1]) / (1 / 60**2))

    velocities: list[float] = [velocity for velocity, *data in
                               instantaneous_velocity(list(data), observe_at_n_moments=POINTS_OF_OBSERVATION,
                                                      derivative_delta_divisor=60.0)]

    accelerations: list[float] = [acceleration for acceleration, *data in
                                  instantaneous_velocity(velocities, observe_at_n_moments=POINTS_OF_OBSERVATION - 1,
                                                         derivative_delta_divisor=60.0)]

    print(f"\n\n\n{len(velocities)} slopes:")
    for slope in velocities:
        print(f"| slope is {slope:.3f}")

    print("\n" * 5)

    for accel in accelerations:
        print(f"| accel is {accel:.3f}")

    pass


if __name__ == '__main__':
    main()
