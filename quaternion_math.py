import numpy as np

def quat2rot(q):

    R = np.array([
        [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
        [2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])],
        [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)]
    ])
    
    return R

def rot2quat(R):

    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        q0 = 0.25 * S
        q1 = (R[2, 1] - R[1, 2]) / S
        q2 = (R[0, 2] - R[2, 0]) / S
        q3 = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q0 = (R[2, 1] - R[1, 2]) / S
        q1 = 0.25 * S
        q2 = (R[0, 1] + R[1, 0]) / S
        q3 = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q0 = (R[0, 2] - R[2, 0]) / S
        q1 = (R[0, 1] + R[1, 0]) / S
        q2 = 0.25 * S
        q3 = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q0 = (R[1, 0] - R[0, 1]) / S
        q1 = (R[0, 2] + R[2, 0]) / S
        q2 = (R[1, 2] + R[2, 1]) / S
        q3 = 0.25 * S

    return np.array([q0, q1, q2, q3])

def conjugate(q):

    return np.array([q[0], -q[1], -q[2], -q[3]])