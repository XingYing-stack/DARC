import numpy as np
import matplotlib.pyplot as plt

TARGET_SOLVER_ACCURACY = {1: 0.5, 2: 0.3, 3: 0.1}

def difficulty_reward(solver_score, difficulty_id):
    if solver_score is None or solver_score < 0:
        return -1.0
    if difficulty_id not in TARGET_SOLVER_ACCURACY:
        return -1.0
    if solver_score == 0:
        return -0.1

    target = float(TARGET_SOLVER_ACCURACY[difficulty_id])
    gap = solver_score - target
    return 1.0 - gap * gap / max(target * (1 - target), 1e-6)

xs = np.linspace(0, 1, 501)

plt.figure(figsize=(6,4))
for did in [3, 2, 1]:   # 3→0.1, 2→0.3, 1→0.5
    ys = [difficulty_reward(x, did) for x in xs]
    plt.plot(xs, ys, label=f"target={TARGET_SOLVER_ACCURACY[did]}")

plt.axhline(0, linestyle="--")
plt.xlabel("solver_score")
plt.ylabel("reward")
plt.legend()
plt.tight_layout()
plt.show()
