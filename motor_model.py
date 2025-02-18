import numpy as np
from scipy.integrate import solve_ivp

class Motor:
    def __init__(self, params):

        self.rpm = np.zeros(4)
        self.rpm_desired = np.zeros(4)
        self.params = params


    def control(self, thrust, torque):
        
        # Motor parameters
        ct = self.params["thrust_coeff"]
        cq = self.params["moment_scale"]
        l = self.params["moment_arm_length"]

        # RPM limits
        rpm_max = 10000
        rpm_min = 1000

        # Calculating desired RPM to send to motor
        A = np.array([[ct, ct, ct, ct],
                     [0, l*ct, 0, -l*ct],
                     [-l*ct, 0, l*ct, 0],
                     [-cq, cq, -cq, cq]])
        
        b = np.array((thrust, *torque))

        self.rpm_desired = np.sqrt(np.maximum(np.linalg.solve(A, b), 0))

        # Applying RPM limits
        self.rpm = np.clip(self.rpm, rpm_min, rpm_max)
        self.rpm_desired = np.clip(self.rpm_desired, rpm_min, rpm_max)

        # Calculate thrust and torque produced by the motor
        u = A @ self.rpm**2
        thrust_motor = u[0]
        torque_motor = u[1:4]
        
        return thrust_motor, torque_motor

    
    def motor_dynamics(self, t, rpm):
     
     km = self.params["motor_constant"]

     return km * (self.rpm_desired - rpm)
    

    def update(self, dt):

        # Rotational dynamics solver
        solution = solve_ivp(
            self.motor_dynamics,
            (0.0, dt),
            self.rpm,
            t_eval=[dt],
            method="RK45"
        )

        # Update
        self.rpm = solution.y[:, -1]


    def get_rpm_data(self):
        return self.rpm, self.rpm_desired