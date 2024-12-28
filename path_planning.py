import numpy as np

def waypoint_discretize(dt):

    # Waypoints in the format of [x position, y position, z position, psi heading]
    waypoints = np.array(
                        [[0, 0, 0, 0],
                         [0, 0, 10, 0],
                         [10, 0, 10, 0],
                         [10, 10, 10, 0],
                         [0, 10, 10, 0],
                         [0, 0, 10, 0],
                         [0, 0, 5, 0]]
                        ).T
    
    # Time to reach each waypoint
    waypoint_times = np.array([0, 5, 10, 15, 20, 25, 30])

    # Discretizing waypoints
    current_waypoint = 0
    path = np.zeros((16, int(waypoint_times[-1]/dt)+1))
    for i in range(int(waypoint_times[-1]/dt)+1):
        if i*dt <= waypoint_times[current_waypoint]:
            psi = waypoints[3, current_waypoint]
            path[4:7, i] = waypoints[0:3, current_waypoint]
            path[0:4, i] = np.array([np.cos(psi/2), 0, 0, np.sin(psi/2)])
        else:
            current_waypoint += 1
            psi = waypoints[3, current_waypoint]
            path[4:7, i] = waypoints[0:3, current_waypoint]
            path[0:4, i] = np.array([np.cos(psi/2), 0, 0, np.sin(psi/2)])

    return path