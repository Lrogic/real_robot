"""SAM 3D single-view vs MV-SAM3D multi-view comparison on D405 captures.

1. capture.py      (vision env, at the arm)  guide the camera to N views, Enter captures each, then mask them
2. reconstruct.py  (sam3d env)               MV-SAM3D on all views + upstream SAM 3D on each view alone
3. compare.py      (sam3d env)               rendered-mesh grid and metrics (reconstruct.py runs it at the end)

See each script's docstring for the commands and the session folder layout.
"""
