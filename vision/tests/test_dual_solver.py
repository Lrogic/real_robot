"""Synthetic checks of the two-arm hand-eye solver (calibration/dual_solver.py).

    cd vision && /opt/conda-envs/vision/bin/python -m unittest tests.test_dual_solver -v
"""

import unittest

import cv2
import numpy as np

from calibration.board import BoardConfig, board_corners_m, load_config, make_board
from calibration.dual_solver import (Obs, initial_guess, initial_guess_fixed_mount, refine, sample_errors,
                                     solve_axzb, summarize)
from calibration.transforms import inv_T, make_T, rotation_between_deg

K = np.array([[655.0, 0, 640.0], [0, 655.0, 360.0], [0, 0, 1]])
DIST = np.array([-0.05, 0.056, 0.001, 0.0007, -0.018])


def rot(rx=0.0, ry=0.0, rz=0.0):
    return cv2.Rodrigues(np.radians([rx, ry, rz]))[0]


def scene(rng, stations=6, per_station=5, noise_px=0.0):
    board_pts = board_corners_m(make_board(BoardConfig.from_dict(load_config()["board"])))
    X = make_T(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float) @ rot(2, -3, 1), [0.03, 0.01, 0.05])
    Z = make_T(rot(0, 0, 178), [1.05, 0.02, 0.01])
    Y = make_T(rot(90, 0, -90) @ rot(3, 2, 1), [0.17, 0.0, 0.02])
    obs = []
    for k in range(stations):
        yaw, pitch = rng.uniform(-30, 30), rng.uniform(-40, 0)
        F = make_T(rot(0, 0, yaw) @ rot(0, -pitch, 0) @ rot(rng.uniform(-10, 10), 0, 0),
                   [rng.uniform(0.3, 0.55), rng.uniform(-0.1, 0.1), rng.uniform(0.1, 0.3)])
        for _ in range(per_station):
            C = make_T(rot(rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-20, 20)),
                       [rng.uniform(-0.08, 0.0), rng.uniform(-0.06, 0.0), rng.uniform(0.3, 0.5)])
            E = Z @ F @ X @ C @ inv_T(Y)
            px, _ = cv2.projectPoints(board_pts, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], K, DIST)
            px = px.reshape(-1, 2) + rng.normal(0, noise_px, (len(board_pts), 2))
            obs.append(Obs(id=len(obs), station=k, E=E, F=F, K=K, dist=DIST, corners_px=px,
                           corner_ids=np.arange(len(board_pts)), C=C))
    return obs, X, Z, Y, board_pts


def pose_error(A, B):
    return rotation_between_deg(A[:3, :3], B[:3, :3]), np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000


class TestDualSolver(unittest.TestCase):
    def test_axzb_exact(self):
        rng = np.random.default_rng(0)
        X = make_T(rot(10, -20, 30), [0.1, -0.2, 0.3])
        Z = make_T(rot(-5, 40, 12), [0.5, 0.1, -0.2])
        Bs = [make_T(rot(*rng.uniform(-60, 60, 3)), rng.uniform(-0.5, 0.5, 3)) for _ in range(6)]
        As = [Z @ B @ inv_T(X) for B in Bs]
        Xs, Zs = solve_axzb(As, Bs)
        for est, true in ((Xs, X), (Zs, Z)):
            r, t = pose_error(est, true)
            self.assertLess(r, 1e-4)              # arccos near 1 resolves only ~1e-6 deg
            self.assertLess(t, 1e-6)

    def test_noise_free_recovery(self):
        obs, X, Z, Y, pts = scene(np.random.default_rng(1))
        X0, Z0, Y0, report = initial_guess(obs)
        self.assertTrue(all(r["used"] for r in report.values()))
        Xr, Zr, Yr, info = refine(obs, X0, Z0, Y0, pts)
        for est, true in ((Xr, X), (Zr, Z), (Yr, Y)):
            r, t = pose_error(est, true)
            self.assertLess(r, 1e-4)
            self.assertLess(t, 1e-3)

    def test_noisy_recovery_and_errors(self):
        obs, X, Z, Y, pts = scene(np.random.default_rng(2), stations=8, per_station=5, noise_px=0.2)
        for o in obs:                                   # PnP pose from the noisy corners, as in real data
            ok, rv, tv = cv2.solvePnP(pts[o.corner_ids], o.corners_px, K, DIST)
            o.C = make_T(cv2.Rodrigues(rv)[0], tv.ravel())
        Xr, Zr, Yr, _ = refine(obs, *initial_guess(obs)[:3], pts)
        for est, true in ((Xr, X), (Zr, Z), (Yr, Y)):
            r, t = pose_error(est, true)
            self.assertLess(r, 0.1)
            self.assertLess(t, 1.0)
        s = summarize([sample_errors(o, Xr, Zr, Yr, pts) for o in obs])
        self.assertLess(s["reproj_rms_px"], 0.4)

    def test_fixed_mount_recovers_base_and_board(self):
        obs, X, Z, Y, pts = scene(np.random.default_rng(4), stations=1, per_station=6)
        Z0, Y0, report = initial_guess_fixed_mount(obs, X)                # one station is enough with X known
        _, Zr, Yr, info = refine(obs, X, Z0, Y0, pts, fix_X=True)
        self.assertTrue(info["fixed_mount"])
        for est, true in ((Zr, Z), (Yr, Y)):
            r, t = pose_error(est, true)
            self.assertLess(r, 1e-4)
            self.assertLess(t, 1e-3)

    def test_too_few_stations(self):
        obs, *_ = scene(np.random.default_rng(3), stations=2)
        with self.assertRaises(ValueError):
            initial_guess(obs)


if __name__ == "__main__":
    unittest.main()
