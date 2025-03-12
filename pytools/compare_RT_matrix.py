import numpy as np
import math

def rotation_error(R1, R2):
    """Calculates rotation error between two rotation matrices."""
    R = np.dot(R1, R2.T)
    trace = np.trace(R)
    angle = math.acos(max(min((trace - 1.0) / 2.0, 1.0), -1.0)) # Clamp to avoid math domain error
    return angle

def translation_error(t1, t2):
    """Calculates translation error between two translation vectors."""
    return np.linalg.norm(t1 - t2)

# matrix_a = np.array([
#     [-0.2560513999839602, 0.9666141473926598, 0.009735020627364019, -0.1612563225471826],
#     [0.5448914270507094, 0.1526428077085601, -0.8244959102272665, 0.1228528070700258],
#     [-0.7984553921747258, -0.2058088028127286, -0.5657841667914506, 0.8759705596712852]
# ])
matrix_a = np.loadtxt("pytools/pose_torch.txt")
matrix_a = np.loadtxt("pytools/posecuda.txt")
# matrix_b = np.array([
#     [-0.56799543, -0.82298964,  0.00831775, -0.10675355],
#     [-0.46207808,  0.31051268, -0.83070197,  0.16739011],
#     [ 0.68107634, -0.47567842, -0.55665517,  0.82034377]
# ])
matrix_b = np.loadtxt("pytools/pose.txt")
R_a = matrix_a[:, :3]
t_a = matrix_a[:, 3]
R_b = matrix_b[:, :3]
t_b = matrix_b[:, 3]

rot_error = rotation_error(R_a, R_b)
trans_error = translation_error(t_a, t_b)

print(f"Rotation Error (radians): {rot_error}")
print(f"Rotation Error (degrees): {np.degrees(rot_error)}")
print(f"Translation Error: {trans_error}")