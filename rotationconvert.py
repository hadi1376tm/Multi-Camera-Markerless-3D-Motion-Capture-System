import numpy as np

# Define the rotation matrix
rotation_matrix = np.array([
    [0.7817921541, 0.001320362326, -0.6235377169],
    [0.04571456871, 0.9971852519, 0.05942854134],
    [0.6218610825, -0.07496552515, 0.7795312464]
    ])

# Convert rotation matrix to Euler angles
euler_angles = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])), np.degrees(np.arcsin(-rotation_matrix[2, 0])), np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))

print("Euler Angles (in degrees):", euler_angles)


import math

def rotation_matrix_to_euler(matrix):
    roll = math.atan2(matrix[2][1], matrix[2][2])
    pitch = math.atan2(-matrix[2][0], math.sqrt(matrix[2][1] ** 2 + matrix[2][2] ** 2))
    yaw = math.atan2(matrix[1][0], matrix[0][0])
    return [roll, pitch, yaw]

# Convert the provided rotation matrix to Euler angles
rotation_angles = rotation_matrix_to_euler([
    [0.7817921541, 0.001320362326, -0.6235377169],
    [0.04571456871, 0.9971852519, 0.05942854134],
    [0.6218610825, -0.07496552515, 0.7795312464]
])

print(rotation_angles)