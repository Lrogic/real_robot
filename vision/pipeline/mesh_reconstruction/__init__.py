"""Textured object meshes from MV-SAM3D, made in the setup stage ("Create mesh" on the mesh screen).

create_mesh.py  (vision env) moves the camera arm to 3 views, masks them, runs worker.py, saves the mesh
views.py        view planning around the object's masked depth points
worker.py       (sam3d env, MV-SAM3D checkout) reconstruction, texture baking, preview render
config.yaml     view geometry, arm goal time, model paths
"""
