import numpy as np

def waypoint_discretize(waypoints, waypoint_times, dt):
    
    # Discretizing waypoints
    current_waypoint = 0
    path = np.zeros((13, int(waypoint_times[-1]/dt)+1))
    for i in range(int(waypoint_times[-1]/dt)+1):
        if i*dt <= waypoint_times[current_waypoint]:
            psi = waypoints[3, current_waypoint]
            path[0:3, i] = waypoints[0:3, current_waypoint]
            path[6:10, i] = np.array([np.cos(psi/2), 0, 0, np.sin(psi/2)])
        else:
            current_waypoint += 1
            psi = waypoints[3, current_waypoint]
            path[0:3, i] = waypoints[0:3, current_waypoint]
            path[6:10, i] = np.array([np.cos(psi/2), 0, 0, np.sin(psi/2)])

    return path