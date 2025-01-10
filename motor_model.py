import numpy as np

class Motor:
    def __init__(self, rpm_init):
        self.rpm

    def control(self, thrust, torque, params):
        
        # Motor parameters
        ct = params["thrust_coeff"]
        cq = params["moment_scale"]
        l = params["moment_arm_length"]

        # RPM limits
        rpm_max = 5000
        rpm_min = 500

        # Calculating desired RPM to send to motor
        A = np.array([ct, ct, ct, ct],
                     [0, l*ct, 0, -l*ct],
                     [-l*ct, 0, l*ct, 0],
                     [-cq, cq, -cq, cq])
        
        b = np.concatenate((thrust, torque))

        rpm_desired = np.sqrt(np.max(np.linalg.solve(A, b), 0))

        # Applying RPM limits
        rpm_desired = np.min(np.max(rpm_desired, rpm_min), rpm_max)

        # Calculate thrust and torque produced by the motor
        thrust_motor = ct * np.sum(rpm_desired**2)
        torque_motor = np.array([l*ct*(rpm_desired[2]**2 - rpm_desired[4]**2),
                                 l*ct*(rpm_desired[3]**2 - rpm_desired[1]**2),
                                 cq*(-rpm_desired[1]**2 + rpm_desired[2]**2 - rpm_desired[3]**2 + rpm_desired[4]**2)])
        
        return thrust_motor, torque_motor

    
    def update(self):
        pass
    
    def get_rpm(self):
        return self.rpm