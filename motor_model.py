import numpy as np
from scipy.integrate import solve_ivp
import dynamics

class Motor:
    def __init__(self):
        self.rpm = np.zeros(4)
        self.rpm_desired = np.zeros(4)

    def control(self, thrust, torque, params):
        
        # Motor parameters
        ct = params["thrust_coeff"]
        cq = params["moment_scale"]
        l = params["moment_arm_length"]

        # RPM limits
        rpm_max = 1000
        rpm_min = 0

        # Calculating desired RPM to send to motor
        A = np.array([[ct, ct, ct, ct],
                     [0, l*ct, 0, -l*ct],
                     [-l*ct, 0, l*ct, 0],
                     [-cq, cq, -cq, cq]])
        
        b = np.concatenate(([thrust], torque))
        self.rpm_desired = np.sqrt(np.maximum(np.linalg.solve(A, b), 0))

        # Applying RPM limits
        self.rpm = np.clip(self.rpm, rpm_min, rpm_max)
        self.rpm_desired = np.clip(self.rpm_desired, rpm_min, rpm_max)

        # Calculate thrust and torque produced by the motor
        thrust_motor = ct * np.sum(self.rpm**2)
        torque_motor = np.array([l*ct*(self.rpm[1]**2 - self.rpm[3]**2),
                                 l*ct*(self.rpm[2]**2 - self.rpm[0]**2),
                                 cq*(-self.rpm[0]**2 + self.rpm[1]**2 - self.rpm[2]**2 + self.rpm[3]**2)])
        
        return thrust_motor, torque_motor

    
    def update(self, params, dt):

        # Rotational dynamics solver
        solution = solve_ivp(
            dynamics.motor_dynamics,
            (0.0, dt),
            self.rpm,
            args=(self.rpm_desired, params),
            t_eval=[dt],
            method="RK45"
        )

        # Update
        self.rpm = solution.y[:, -1]


    def get_rpm_data(self):
        return self.rpm