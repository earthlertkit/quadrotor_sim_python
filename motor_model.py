import numpy as np
from scipy.integrate import solve_ivp
import dynamics

class Motor:
    def __init__(self):
        self.rpm = 0
        self.rpm_desired = 0

    def control(self, thrust, torque, params):
        
        # Motor parameters
        ct = params["thrust_coeff"]
        cq = params["moment_scale"]
        l = params["moment_arm_length"]

        # RPM limits
        rpm_max = 10
        rpm_min = 0

        # Calculating desired RPM to send to motor
        A = np.array([ct, ct, ct, ct],
                     [0, l*ct, 0, -l*ct],
                     [-l*ct, 0, l*ct, 0],
                     [-cq, cq, -cq, cq])
        
        b = np.concatenate((thrust, torque))

        self.rpm_desired = np.sqrt(np.max(np.linalg.solve(A, b), 0))

        # Applying RPM limits
        self.rpm = np.clip(self.rpm, rpm_min, rpm_max)
        self.rpm_desired = np.clip(self.rpm_desired, rpm_min, rpm_max)

        # Calculate thrust and torque produced by the motor
        thrust_motor = ct * np.sum(self.rpm**2)
        torque_motor = np.array([l*ct*(self.rpm[2]**2 - self.rpm[4]**2),
                                 l*ct*(self.rpm[3]**2 - self.rpm[1]**2),
                                 cq*(-self.rpm[1]**2 + self.rpm[2]**2 - self.rpm[3]**2 + self.rpm[4]**2)])
        
        return thrust_motor, torque_motor

    
    def update(self, params):

        # Rotational dynamics solver
        solution = solve_ivp(
            dynamics.rotational_dynamics,
            (0.0, self.dt),
            self.rpm,
            args=(self.rpm_desired, params),
            t_eval=[self.dt],
            method="RK45"
        )

        # Update
        self.rpm = solution.y[:, -1]