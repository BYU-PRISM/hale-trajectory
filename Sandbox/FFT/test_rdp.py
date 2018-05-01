# -*- coding: utf-8 -*-

from rdp import rdp
import pandas as pd

def angle(directions):
    """Return the angle between vectors
    """
    vec2 = directions[1:]
    vec1 = directions[:-1]

    norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
    norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
    cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)   
    return np.arccos(cos)

data = pd.read_excel('dataLong.xlsx')

trajectory = data[['x','y']].as_matrix()

# Build simplified (approximated) trajectory
# using RDP algorithm.
simplified_trajectory = rdp(trajectory, epsilon=200)
sx, sy = simplified_trajectory.T

# Visualize trajectory and its simplified version.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, 'r--', label='trajectory')
ax.plot(sx, sy, 'b-', label='simplified trajectory')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc='best')

# Define a minimum angle to treat change in direction
# as significant (valuable turning point).
min_angle = np.pi / 2.5

# Compute the direction vectors on the simplified_trajectory.
directions = np.diff(simplified_trajectory, axis=0)
theta = angle(directions)

# Select the index of the points with the greatest theta.
# Large theta is associated with greatest change in direction.
idx = np.where(theta > min_angle)[0] + 1

# Visualize valuable turning points on the simplified trjectory.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sx, sy, 'gx-', label='simplified trajectory')
ax.plot(sx[idx], sy[idx], 'ro', markersize = 7, label='turning points')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(loc='best')