import numpy as np

REFERENCE_IMPLEMENTATION = True


if REFERENCE_IMPLEMENTATION:
    def predict(x, P, F, Q):
        x = F @ x
        P = F @ P @ F.T + Q
        return x, P

    def update(x, P, z, R, H):
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(len(x)) - K @ H) @ P
        return x, P
else:
    from filterpy.kalman import update, predict