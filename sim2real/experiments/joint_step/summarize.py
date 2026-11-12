"""Tables behind docs/sim2real_control.md. Environment: wxai_mjlab. Run from this directory.

    python summarize.py

Reads the two hardware sessions and the sim sessions in results/ (the .npz sample logs are
git-ignored; rerun run_hardware.py / run_sim.py to regenerate them). The measure throughout is
the fraction of the commanded step a joint covers in 50 ms (step, ramp) or per 50 ms policy
tick (relative), from analyze.Session.metrics.

  A   hardware, per joint: that fraction at the ready pose / at the reach pose
  A2  the same for the goal-time-0 step test, split by step size
  B   mean |hardware - sim| of that fraction per joint, for the training model -> the model
      with fitted joint friction, a 6 ms command delay and the goal-time command model;
      then all joints together, per pose
"""
import numpy as np

from analyze import Session
hw = Session('results/hardware_20261111_142208.npz'); hw1 = Session('results/hardware_20261111_151626.npz')
sd = Session('results/sim_training_model.npz'); sf = Session('results/sim_friction_delay.npz'); sf1 = Session('results/sim_friction_delay_once.npz')
def m(s, n):
    r = s.metrics(n); return r.get('per_tick', r.get('at_50ms'))
def groups(s):
    g = {}
    for n, t in s.trials.items():
        g.setdefault((t['kind'], t['goal_time'], t['joint'], t['pose']), []).append(n)
    return g
print('A) hardware: fraction of commanded step moved in 50 ms (step/ramp) or per tick (relative); ready / reach')
for S in (hw, hw1):
    g = groups(S)
    for kind, gt in sorted({k[:2] for k in g}):
        row = []
        for j in range(6):
            a = np.mean([m(S, n) for n in g[(kind, gt, j, 'ready')]]); b = np.mean([m(S, n) for n in g[(kind, gt, j, 'reach')]])
            row.append(f'{a:.2f}/{b:.2f}')
        print(f'  {kind:14s} gt{gt*1e3:3.0f}  ' + '  '.join(row))
print('A2) same, split by amplitude, step gt0 only')
for amp in (0.005, 0.01, 0.02, 0.04):
    row = []
    for j in range(6):
        v = [np.mean([m(hw, n) for n, t in hw.trials.items() if t['kind']=='step' and t['goal_time']==0 and t['joint']==j and t['pose']==p and abs(abs(t['amplitude'])-amp)<1e-9]) for p in ('ready','reach')]
        row.append(f'{v[0]:.2f}/{v[1]:.2f}')
    print(f'  {amp*1e3:4.0f} mrad  ' + '  '.join(row))
print('B) mean |real - sim| of that fraction: training model -> friction+delay(+exp_ff), per joint; then all-joint by pose')
for H, D, F in ((hw, sd, sf), (hw1, None, sf1)):
    g = groups(H)
    for kind, gt in sorted({k[:2] for k in g}):
        out = []
        for j in range(6):
            ns = g[(kind, gt, j, 'ready')] + g[(kind, gt, j, 'reach')]
            d = np.mean([abs(m(H, n) - m(D, n)) for n in ns]) if D else float('nan')
            f = np.mean([abs(m(H, n) - m(F, n)) for n in ns])
            out.append(f'{d:.2f}>{f:.2f}')
        bp = []
        for p in ('ready', 'reach'):
            ns = sum((g[(kind, gt, j, p)] for j in range(6)), [])
            d = np.mean([abs(m(H, n) - m(D, n)) for n in ns]) if D else float('nan')
            bp.append(f"{p} {d:.2f}>{np.mean([abs(m(H, n) - m(F, n)) for n in ns]):.2f}")
        print(f'  {kind:14s} gt{gt*1e3:3.0f}  ' + '  '.join(out) + '   | ' + '  '.join(bp))
