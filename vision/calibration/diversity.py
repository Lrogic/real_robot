"""Motion-diversity and consistency analysis of collected (E_i, C_i) pairs.

A_ij = E_i @ inv(E_j) and B_ij = C_i @ inv(C_j) are conjugate (A_ij @ X = X @ B_ij), so their
rotation angles must agree whatever X is: a mismatch points to a slipping board, a bent
mount or a bad detection. Rotation axes of A_ij that are all (nearly) parallel leave the
translation of X along that axis unobservable, however many poses are added.
"""

import itertools

import numpy as np

from .transforms import inv_T, rotation_axis_angle, rotation_between_deg


def analyze(Es, Cs, metrics, div: dict, ids=None) -> dict:
    n = len(Es)
    ids = list(ids) if ids is not None else list(range(n))
    lo, hi = div["target_samples"]
    warnings = []
    stats = {"num_samples": n}
    if n < lo:
        warnings.append(f"{lo - n} more poses recommended (target {lo}-{hi})")

    if n:
        def span(key):
            v = [m[key] for m in metrics if key in m]
            return (min(v), max(v)) if v else (0.0, 0.0)

        tx, ty, ip, dist = span("tilt_about_x_deg"), span("tilt_about_y_deg"), span("inplane_deg"), \
            span("distance_m")
        stats.update(tilt_about_x_deg=tx, tilt_about_y_deg=ty, inplane_deg=ip, distance_m=dist)
        t_min = div["tilt_target_deg"][0]
        if n >= 3:
            if max(abs(tx[0]), abs(tx[1])) < t_min:
                warnings.append(f"add tilts about the camera X axis (board top/bottom toward camera, >{t_min:g} deg)")
            if max(abs(ty[0]), abs(ty[1])) < t_min:
                warnings.append(f"add tilts about the camera Y axis (board left/right toward camera, >{t_min:g} deg)")
            if ip[1] - ip[0] < 30:
                warnings.append("add in-plane rotations of the board (spin it about its normal)")
            if dist[1] - dist[0] < 0.05:
                warnings.append("vary the board distance to the camera")

    pairs, axes, weights, mismatch = [], [], [], {i: [] for i in range(n)}
    for i, j in itertools.combinations(range(n), 2):
        A = Es[i] @ inv_T(Es[j])
        B = Cs[i] @ inv_T(Cs[j])
        axis, ang_a = rotation_axis_angle(A[:3, :3])
        _, ang_b = rotation_axis_angle(B[:3, :3])
        pairs.append(ang_a)
        d = abs(ang_a - ang_b)
        mismatch[i].append(d)
        mismatch[j].append(d)
        if ang_a >= div["min_pair_rotation_deg"]:
            axes.append(axis)
            weights.append(ang_a)

    if pairs:
        small = float(np.mean(np.array(pairs) < div["min_pair_rotation_deg"]))
        stats["fraction_small_pair_rotations"] = small
        stats["max_pair_rotation_deg"] = float(max(pairs))
        if n >= 3 and small > 0.5:
            warnings.append("most pose pairs differ by little rotation; use larger orientation changes")
    if len(axes) >= 2:
        a = np.asarray(axes)
        M = (a * np.asarray(weights)[:, None]).T @ a
        ev = np.sort(np.linalg.eigvalsh(M))[::-1]
        stats["axis_spread"] = float(ev[1] / ev[0]) if ev[0] > 0 else 0.0
        stats["axis_spread_3"] = float(ev[2] / ev[0]) if ev[0] > 0 else 0.0
        if n >= 3 and stats["axis_spread"] < div["min_axis_spread"]:
            warnings.append("rotations are mostly about ONE axis: add rotations about a different axis "
                            "(more poses about the same axis do not remove the ambiguity)")
    elif n >= 3:
        warnings.append("rotations too small to judge axis diversity")

    per_sample = {ids[i]: float(np.median(v)) for i, v in mismatch.items() if v}
    stats["angle_mismatch_deg"] = per_sample
    bad = [k for k, v in per_sample.items() if v > div["max_angle_mismatch_deg"]]
    if bad:
        warnings.append(f"inconsistent samples {bad}: robot vs camera rotation disagree "
                        f"(>{div['max_angle_mismatch_deg']:g} deg) - board slipped/bent or bad detection?")
    return {"stats": stats, "warnings": warnings, "inconsistent": bad}


def nearest_duplicate(Es, div: dict):
    """Index of an earlier sample the last one nearly duplicates, or None."""
    if len(Es) < 2:
        return None
    last = Es[-1]
    for k, E in enumerate(Es[:-1]):
        if (rotation_between_deg(E[:3, :3], last[:3, :3]) < div["similar_rotation_deg"]
                and np.linalg.norm(E[:3, 3] - last[:3, 3]) < div["similar_translation_m"]):
            return k
    return None


def summary_lines(result: dict) -> list:
    s = result["stats"]
    lines = [f"samples: {s['num_samples']}"]
    if "tilt_about_x_deg" in s:
        lines.append("tilt X {:+.0f}..{:+.0f}  Y {:+.0f}..{:+.0f} deg".format(*s["tilt_about_x_deg"],
                                                                         *s["tilt_about_y_deg"]))
        lines.append("in-plane {:+.0f}..{:+.0f} deg  dist {:.2f}..{:.2f} m".format(*s["inplane_deg"],
                                                                                  *s["distance_m"]))
    if "axis_spread" in s:
        lines.append(f"rotation-axis spread {s['axis_spread']:.2f} / {s['axis_spread_3']:.2f}")
    return lines
