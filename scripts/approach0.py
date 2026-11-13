"""
Approach 0: analytical first estimates of the overtaking boundary (proposal, Section 5.1).

Three tables, to compare later with the simulated overtaking maps:
1. Passing time tau = (g + L_O + L_E + m) / dv for a grip advantage and for a speed advantage.
2. Braking: distance, later braking point, and what braking later really gains.
3. First-guess boundary: largest initial time gap that each advantage could make up.

    python scripts/approach0.py                          # hairpin (R = 15 m, 180 deg)
    python scripts/approach0.py --radius 40 --angle 90   # 90 deg corner (numbers of the proposal)
    python scripts/approach0.py --csv results/approach0.csv

Model: point mass on the centreline, both cars on the same line, the opponent never in the way.
The results are therefore optimistic: they say where an overtake cannot happen.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from overtake.speed_profile import three_pass, time_stamps
from overtake.track import CornerTrack
from overtake.types import VehicleParams


# ------------------------------------ Formulas ---------------------------------------
def passing_time(gap, dv, length_opponent, length_ego, margin):
    """
    Time needed to pass with a constant speed advantage.

    Args:
        gap: float. Bumper-to-bumper gap at the start [m].
        dv: float. Speed advantage of the ego [m/s].
        length_opponent, length_ego: float. Car lengths [m].
        margin: float. Extra distance wanted once ahead [m].

    Returns:
        float. tau = (gap + L_O + L_E + margin) / dv [s]; infinite if dv <= 0.
    """
    return (gap + length_opponent + length_ego + margin) / dv if dv > 0 else np.inf


def apex_speed(a_y_max, radius):
    """ Highest speed in a corner of the given radius [m/s]: all the grip goes to cornering. """
    return np.sqrt(a_y_max * radius)


def braking_distance(v_in, v_apex, a_x_max):
    """ Distance needed to slow from v_in to v_apex on a straight with full braking [m]. """
    return (v_in ** 2 - v_apex ** 2) / (2.0 * a_x_max)


def late_braking_gain(v_in, v_apex, a_opponent, a_ego):
    """
    What the ego gains on the opponent by braking later, both arriving at v_in.

    Braking D metres later does not put the ego D metres ahead. While the opponent already
    brakes, the ego closes in; once the ego brakes harder, the opponent comes back. Both
    reach the same apex speed, so only a small time gain remains.

    Args:
        v_in: float. Speed of both cars before braking [m/s].
        v_apex: float. Speed of both cars at the end of braking [m/s].
        a_opponent, a_ego: float. Braking decelerations [m/s^2], a_ego >= a_opponent.

    Returns:
        later: float. How much later the ego brakes [m].
        peak: float. Largest distance gained during the braking phase [m].
        time_gain: float. Time gained at the end of braking [s].
    """
    later = braking_distance(v_in, v_apex, a_opponent) - braking_distance(v_in, v_apex, a_ego)
    t_alone = later / v_in                                   # the opponent brakes, the ego not yet
    dv = a_opponent * t_alone                                # speed difference when the ego starts braking
    t_both = (v_in - v_apex) / a_ego                         # duration of the ego's braking
    t_peak = t_both if a_ego <= a_opponent else min(dv / (a_ego - a_opponent), t_both)
    peak = 0.5 * a_opponent * t_alone ** 2 + dv * t_peak - 0.5 * (a_ego - a_opponent) * t_peak ** 2
    time_gain = (v_in - v_apex) / a_opponent - (t_alone + t_both)
    return later, peak, time_gain


def time_advantage(track, opponent, ego, v_in, dv0=0.0):
    """
    Time the ego gains on the opponent when each drives the centreline alone (three-pass profile).

    Args:
        track: CornerTrack.
        opponent, ego: VehicleParams.
        v_in: float. Entry speed of the opponent [m/s].
        dv0: float. Extra entry speed of the ego [m/s].

    Returns:
        (to_corner, in_corner, on_exit): time gained on the entry straight, in the arc and on
        the exit straight [s]. Their sum is the gain over the whole segment.
    """
    times = []
    for params, v_start in ((opponent, v_in), (ego, v_in + dv0)):
        v = three_pass(track.s, track.kappa, params, v_start)
        t = time_stamps(track.s, v)
        times.append(np.interp([track.corner_start, track.corner_end, track.length], track.s, t))
    gain = times[0] - times[1]                               # cumulative gain at the three points
    return gain[0], gain[1] - gain[0], gain[2] - gain[1]


# ------------------------------------ Tables ---------------------------------------
def print_table(title, header, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *rows)]
    print(f"\n{title}")
    print("  ".join(f"{name:>{w}}" for name, w in zip(header, widths)))
    for row in rows:
        print("  ".join(f"{cell:>{w}}" for cell, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--l-in", type=float, default=250.0)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--angle", type=float, default=180.0, help="corner angle [deg], positive = left")
    parser.add_argument("--l-out", type=float, default=150.0)
    parser.add_argument("--v-in", type=float, default=60.0, help="approach speed [m/s]")
    parser.add_argument("--gap", type=float, default=5.0, help="initial bumper-to-bumper gap g [m]")
    parser.add_argument("--margin", type=float, default=1.0, help="distance wanted once ahead m [m]")
    parser.add_argument("--tau-min", type=float, default=0.1, help="time margin required at the finish [s]")
    parser.add_argument("--csv", type=Path, default=None, help="save the boundary table to this file")
    args = parser.parse_args()

    p = VehicleParams()
    track = CornerTrack(args.l_in, args.radius, args.angle, args.l_out)
    v_apex = apex_speed(p.a_y_max, track.radius)
    corner_time = (track.corner_end - track.corner_start) / v_apex
    d_brake = braking_distance(args.v_in, v_apex, p.a_x_max)
    clear = args.gap + p.length + args.margin                # distance to gain to be clear ahead
    print(f"Corner: R = {track.radius:g} m, {track.angle_deg:g} deg, straights {track.l_in:g} m and "
          f"{track.l_out:g} m. Approach speed {args.v_in:g} m/s.")
    print(f"Opponent: apex speed {v_apex:.1f} m/s, {corner_time:.2f} s in the corner, "
          f"braking distance {d_brake:.1f} m.")

    # 1. Passing time.
    rows = []
    for r_y in (1.05, 1.10, 1.15):
        dv = apex_speed(p.a_y_max * r_y, track.radius) - v_apex
        tau = passing_time(args.gap, dv, p.length, p.length, args.margin)
        rows.append((f"grip x {r_y:.2f}", f"{dv:.2f}", f"{tau:.1f}", f"{corner_time:.2f} (corner)",
                     "yes" if tau <= corner_time else "no"))
    straight_time = (track.l_in - d_brake) / args.v_in       # time before the braking point
    for dv in (1.0, 2.0, 5.0):
        tau = passing_time(args.gap, dv, p.length, p.length, args.margin)
        rows.append((f"speed + {dv:g} m/s", f"{dv:.2f}", f"{tau:.1f}", f"{straight_time:.2f} (straight)",
                     "yes" if tau <= straight_time else "no"))
    print_table(f"1. Passing time tau = (g + L_O + L_E + m) / dv, with g = {args.gap:g} m, m = {args.margin:g} m",
                ("advantage", "dv [m/s]", "tau [s]", "time available [s]", "enough?"), rows)

    # 2. Braking.
    rows = []
    for r_x in (1.05, 1.10, 1.20):
        later, peak, gain = late_braking_gain(args.v_in, v_apex, p.a_x_max, p.a_x_max * r_x)
        rows.append((f"brakes x {r_x:.2f}", f"{d_brake - later:.1f}", f"{later:.1f}", f"{peak:.1f}",
                     f"{gain:.3f}", f"{clear:.1f}", "yes" if peak >= clear else "no"))
    print_table(f"2. Braking from {args.v_in:g} m/s to the apex speed (opponent: {d_brake:.1f} m)",
                ("advantage", "distance [m]", "brakes later by [m]", "peak gain [m]", "time gain [s]",
                 "needed [m]", "enough?"), rows)

    # 3. First-guess boundary in the plane (ratio, initial time gap).
    cases = [("a_y_max", r) for r in (1.05, 1.10, 1.15)] + [("a_x_max", r) for r in (1.05, 1.10, 1.15)] \
        + [("a_engine", r) for r in (1.05, 1.10)] + [("v_max", r) for r in (1.02, 1.05)]
    records = []
    for name, ratio in cases:
        gains = time_advantage(track, p, p.scaled(**{name: ratio}), args.v_in)
        records.append((name, f"x {ratio:.2f}", gains))
    for dv0 in (2.0, 5.0):
        records.append(("initial speed", f"+ {dv0:g} m/s", time_advantage(track, p, p, args.v_in, dv0)))
    rows, csv_rows = [], []
    for name, value, gains in records:
        total = sum(gains)
        tau0_max = total - args.tau_min
        rows.append((name, value, *(f"{g:+.3f}" for g in gains), f"{total:+.3f}",
                     f"{tau0_max:.2f}" if tau0_max > 0 else "none"))
        csv_rows.append((name, value.replace("x ", "").replace("+ ", "").replace(" m/s", ""),
                         *(round(g, 4) for g in gains), round(total, 4), round(max(tau0_max, 0.0), 4)))
    header = ("parameter", "advantage", "entry [s]", "corner [s]", "exit [s]", "total [s]", "max initial gap [s]")
    print_table("3. Time gained by the ego driving alone, and largest initial time gap it could make up "
                f"(total - {args.tau_min:g} s)", header, rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("parameter", "advantage", "gain_entry_s", "gain_corner_s", "gain_exit_s",
                             "gain_total_s", "max_initial_gap_s"))
            writer.writerows(csv_rows)
        print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()
