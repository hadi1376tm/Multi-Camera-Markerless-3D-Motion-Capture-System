import pandas as pd
import numpy as np
import math

def cal3Dangle(p1, p2, p3):
    # calculate vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2
    # calculate dot product of v1 and v2
    dot_product = np.dot(v1, v2)
    # calculate magnitudes of v1 and v2
    v1_magnitude = np.linalg.norm(v1)
    v2_magnitude = np.linalg.norm(v2)
    # calculate angle using dot product formula
    # Calculate the cosine of the angle
    cos_angle = dot_product / (v1_magnitude * v2_magnitude)

    # Ensure the cosine value is within the valid range [-1, 1]
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = math.acos(cos_angle)
    # Ensure angle is positive
    if angle < 0:
        angle = 2 * np.pi + angle
    # Convert angle to degrees
    angle_in_degrees = math.degrees(angle)
    return angle_in_degrees

def calculate_angles(selectedJoints,csvpath):
    # read 3D coordinates from csv file
    print('-Calculating angles')
    df = pd.read_csv(csvpath /'mediapipe_body_3d_xyz.csv' )
    # calculate angle between selected joints for each row
    left_elbow_angles = []
    right_elbow_angles = []
    right_shoulder_angles = []
    left_shoulder_angles = []
    right_hip_angles = []
    left_hip_angles = []
    right_knee_angles = []
    left_knee_angles = []
    right_ankle_angles = []
    left_ankle_angles = []


    # my_dict = {"left_elbow": [], "right_elbow": "value2", "left_shoulder": "value1", "right_shoulder": "value2",
    #            "left_hip": "value1", "right_hip": "value2","left_knee": "value1", "right_knee": "value2"}
    # 
    
    for index, row in df.iterrows():
        
        left_wrist = np.array([row["left_wrist_x"], row["left_wrist_y"], row["left_wrist_z"]])
        right_wrist = np.array([row["right_wrist_x"], row["right_wrist_y"], row["right_wrist_z"]])
        
        left_elbow = np.array([row["left_elbow_x"], row["left_elbow_y"], row["left_elbow_z"]])
        right_elbow = np.array([row["right_elbow_x"], row["right_elbow_y"], row["right_elbow_z"]])
        
        right_shoulder = np.array([row["right_shoulder_x"], row["right_shoulder_y"], row["right_shoulder_z"]])
        left_shoulder = np.array([row["left_shoulder_x"], row["left_shoulder_y"], row["left_shoulder_z"]])
        
        right_hip = np.array([row["right_hip_x"], row["right_hip_y"], row["right_hip_z"]])
        left_hip = np.array([row["left_hip_x"], row["left_hip_y"], row["left_hip_z"]])

        left_knee = np.array([row["left_knee_x"], row["left_knee_y"], row["left_knee_z"]])
        right_knee = np.array([row["right_knee_x"], row["right_knee_y"], row["right_knee_z"]])
        
        left_ankle = np.array([row["left_ankle_x"], row["left_ankle_y"], row["left_ankle_z"]])
        right_ankle = np.array([row["right_ankle_x"], row["right_ankle_y"], row["right_ankle_z"]])

        left_foot_index = np.array([row["left_foot_index_x"], row["left_foot_index_y"], row["left_foot_index_z"]])
        right_foot_index = np.array([row["right_foot_index_x"], row["right_foot_index_y"], row["right_foot_index_z"]])



        angle = cal3Dangle(left_wrist, left_elbow, left_shoulder)
        left_elbow_angles.append(angle)

        angle = cal3Dangle(left_elbow, left_shoulder, left_hip)
        left_shoulder_angles.append(angle)

        angle = cal3Dangle(left_shoulder, left_hip, left_knee)
        left_hip_angles.append(angle)


        angle = cal3Dangle(left_hip, left_knee, left_ankle)
        left_knee_angles.append(angle)

        angle = cal3Dangle(right_wrist, right_elbow, right_shoulder)
        right_elbow_angles.append(angle)

        angle = cal3Dangle(right_elbow, right_shoulder, right_hip)
        right_shoulder_angles.append(angle)

        angle = cal3Dangle(right_shoulder, right_hip, right_knee)
        right_hip_angles.append(angle)

        angle = cal3Dangle(right_hip, right_knee, right_ankle)
        right_knee_angles.append(angle)

        angle = cal3Dangle(right_knee, right_ankle, right_foot_index)
        right_ankle_angles.append(angle)
        
        angle = cal3Dangle(left_knee, left_ankle, left_foot_index)
        left_ankle_angles.append(angle)

    angles = {"left_elbow": left_elbow_angles, "right_elbow": right_elbow_angles, "left_shoulder": left_shoulder_angles,
              "right_shoulder": right_shoulder_angles, "left_hip": left_hip_angles, "right_hip": right_hip_angles,
              "left_knee": left_knee_angles, "right_knee": right_knee_angles,"left_ankle": left_ankle_angles, "right_ankle": right_ankle_angles}

    df = pd.DataFrame(angles)
    df.to_csv(csvpath / 'mediapipe_angles.csv', index=False)


def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_distances(csvpath):
    print('-Calculating distances')
    df = pd.read_csv(csvpath / 'mediapipe_body_3d_xyz.csv')

    ankle_distances = []
    shoulder_distances = []  # New list to store shoulder distances

    for index, row in df.iterrows():
        left_ankle = np.array([row["left_ankle_x"], row["left_ankle_y"], row["left_ankle_z"]])
        right_ankle = np.array([row["right_ankle_x"], row["right_ankle_y"], row["right_ankle_z"]])

        distance_ankle = calculate_distance(left_ankle, right_ankle)
        ankle_distances.append(distance_ankle)

        left_shoulder = np.array([row["left_shoulder_x"], row["left_shoulder_y"], row["left_shoulder_z"]])
        right_shoulder = np.array([row["right_shoulder_x"], row["right_shoulder_y"], row["right_shoulder_z"]])

        distance_shoulder = calculate_distance(left_shoulder, right_shoulder)
        shoulder_distances.append(distance_shoulder)

    df['ankle_distance'] = ankle_distances
    df['shoulder_distance'] = shoulder_distances  # Adding shoulder distances to the DataFrame

    # Save the 'ankle_distance' and 'shoulder_distance' columns to a new CSV file
    df[['ankle_distance', 'shoulder_distance']].to_csv(csvpath / 'mediapipe_distances.csv', index=False)

