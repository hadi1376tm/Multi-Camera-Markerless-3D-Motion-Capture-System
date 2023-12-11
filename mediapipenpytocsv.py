from pathlib import Path
import pandas as pd
import numpy as np
from mediapipe.python.solutions import holistic as mp_holistic

def mediapipetocsv(path):

    mediapipe_3d_npy_path = Path(path)

    mediapipe_3d_frame_trackedPoint_xyz = np.load(str(mediapipe_3d_npy_path))

    print(f'loaded npy data with shape: {mediapipe_3d_frame_trackedPoint_xyz.shape}')

    pose_landmark_names = [landmark.name.lower() for landmark in mp_holistic.PoseLandmark]
    hand_landmark_names = [landmark.name.lower() for landmark in mp_holistic.HandLandmark]

    # get number of points in body, hands, face

    number_of_body_points = len(pose_landmark_names)
    number_of_hand_points = len(hand_landmark_names)

    first_body_marker_index = 0
    last_body_marker_index = number_of_body_points - 1

    first_right_hand_marker_index = last_body_marker_index + 1
    last_right_hand_marker_index = number_of_body_points + number_of_hand_points - 1

    first_left_hand_marker_index = last_right_hand_marker_index + 1
    last_left_hand_marker_index = last_right_hand_marker_index + 1 + number_of_hand_points - 1

    first_face_marker_index = last_left_hand_marker_index + 1
    last_face_marker_index = mediapipe_3d_frame_trackedPoint_xyz.shape[1]

    number_of_face_points = last_face_marker_index - first_face_marker_index



    body_3d_xyz = mediapipe_3d_frame_trackedPoint_xyz[:, first_body_marker_index:last_body_marker_index + 1, :]
    right_hand_3d_xyz = mediapipe_3d_frame_trackedPoint_xyz[:,
                        first_right_hand_marker_index:last_right_hand_marker_index + 1, :]
    left_hand_3d_xyz = mediapipe_3d_frame_trackedPoint_xyz[:, first_left_hand_marker_index:last_left_hand_marker_index + 1,
                       :]
    face_3d_xyz = mediapipe_3d_frame_trackedPoint_xyz[:, first_face_marker_index:last_face_marker_index + 1, :]

    # save broken up npy files
    np.save(str(mediapipe_3d_npy_path.parent / "mediapipe_body_3d_xyz.npy"), body_3d_xyz)
    np.save(str(mediapipe_3d_npy_path.parent / "mediapipe_right_hand_3d_xyz.npy"), right_hand_3d_xyz)
    np.save(str(mediapipe_3d_npy_path.parent / "mediapipe_left_hand_3d_xyz.npy"), left_hand_3d_xyz)
    np.save(str(mediapipe_3d_npy_path.parent / "mediapipe_face_3d_xyz.npy"), face_3d_xyz)

    # create pandas data frame headers

    body_3d_xyz_header = []
    for landmark_name in pose_landmark_names:
        body_3d_xyz_header.append(f"{landmark_name}_x")
        body_3d_xyz_header.append(f"{landmark_name}_y")
        body_3d_xyz_header.append(f"{landmark_name}_z")

    right_hand_3d_xyz_header = []
    for landmark_name in hand_landmark_names:
        right_hand_3d_xyz_header.append(f"right_hand_{landmark_name}_x")
        right_hand_3d_xyz_header.append(f"right_hand_{landmark_name}_y")
        right_hand_3d_xyz_header.append(f"right_hand_{landmark_name}_z")

    left_hand_3d_xyz_header = []
    for landmark_name in hand_landmark_names:
        left_hand_3d_xyz_header.append(f"left_hand_{landmark_name}_x")
        left_hand_3d_xyz_header.append(f"left_hand_{landmark_name}_y")
        left_hand_3d_xyz_header.append(f"left_hand_{landmark_name}_z")

    face_3d_xyz_header = []
    for landmark_number in range(last_face_marker_index - first_face_marker_index):
        face_3d_xyz_header.append(f"face_{str(landmark_number).zfill(4)}_x")
        face_3d_xyz_header.append(f"face_{str(landmark_number).zfill(4)}_y")
        face_3d_xyz_header.append(f"face_{str(landmark_number).zfill(4)}_z")


    number_of_frames = mediapipe_3d_frame_trackedPoint_xyz.shape[0]
    body_flat = body_3d_xyz.reshape(number_of_frames, number_of_body_points * 3)

    body_dataframe = pd.DataFrame(body_flat, columns=body_3d_xyz_header)
    body_dataframe.to_csv(str(mediapipe_3d_npy_path.parent / "mediapipe_body_3d_xyz.csv"), index=False)

    right_hand_flat = right_hand_3d_xyz.reshape(number_of_frames, number_of_hand_points * 3)
    right_hand_dataframe = pd.DataFrame(right_hand_flat, columns=right_hand_3d_xyz_header)
    right_hand_dataframe.to_csv(str(mediapipe_3d_npy_path.parent / "mediapipe_right_hand_3d_xyz.csv"), index=False)

    left_hand_flat = left_hand_3d_xyz.reshape(number_of_frames, number_of_hand_points * 3)
    left_hand_dataframe = pd.DataFrame(left_hand_flat, columns=left_hand_3d_xyz_header)
    left_hand_dataframe.to_csv(str(mediapipe_3d_npy_path.parent / "mediapipe_left_hand_3d_xyz.csv"), index=False)

    face_flat = face_3d_xyz.reshape(number_of_frames, number_of_face_points * 3)
    face_dataframe = pd.DataFrame(face_flat, columns=face_3d_xyz_header)
    face_dataframe.to_csv(str(mediapipe_3d_npy_path.parent / "mediapipe_face_3d_xyz.csv"), index=False)

    pose_landmark_names = [landmark.name.lower() for landmark in mp_holistic.PoseLandmark]
    hand_landmark_names = [landmark.name.lower() for landmark in mp_holistic.HandLandmark]

    names: ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
            'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']
    ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip', 'index_finger_mcp', 'index_finger_pip', 'index_finger_dip',
     'index_finger_tip', 'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
     'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip', 'pinky_mcp', 'pinky_pip', 'pinky_dip',
     'pinky_tip']

    # get number of points in body, hands, face

    number_of_body_points = len(pose_landmark_names)
    number_of_hand_points = len(hand_landmark_names)

    first_body_marker_index = 0
    last_body_marker_index = number_of_body_points - 1

    first_right_hand_marker_index = last_body_marker_index + 1
    last_right_hand_marker_index = number_of_body_points + number_of_hand_points - 1

    first_left_hand_marker_index = last_right_hand_marker_index + 1
    last_left_hand_marker_index = last_right_hand_marker_index + 1 + number_of_hand_points - 1

    first_face_marker_index = last_left_hand_marker_index + 1
    last_face_marker_index = mediapipe_3d_frame_trackedPoint_xyz.shape[1]

    number_of_face_points = last_face_marker_index - first_face_marker_index

    body_3d_xyz = mediapipe_3d_frame_trackedPoint_xyz[:,first_body_marker_index:last_body_marker_index+1,:]
    right_hand_3d_xyz = mediapipe_3d_frame_trackedPoint_xyz[:,first_right_hand_marker_index:last_right_hand_marker_index+1,:]
    left_hand_3d_xyz = mediapipe_3d_frame_trackedPoint_xyz[:,first_left_hand_marker_index:last_left_hand_marker_index+1,:]
    face_3d_xyz = mediapipe_3d_frame_trackedPoint_xyz[:,first_face_marker_index:last_face_marker_index+1,:]

    # save broken up npy files
    np.save(str(mediapipe_3d_npy_path.parent / "mediapipe_body_3d_xyz.npy"), body_3d_xyz)
    np.save(str(mediapipe_3d_npy_path.parent / "mediapipe_right_hand_3d_xyz.npy"), right_hand_3d_xyz)
    np.save(str(mediapipe_3d_npy_path.parent / "mediapipe_left_hand_3d_xyz.npy"), left_hand_3d_xyz)
    np.save(str(mediapipe_3d_npy_path.parent / "mediapipe_face_3d_xyz.npy"), face_3d_xyz)

    # create pandas data frame headers

    body_3d_xyz_header = []
    for landmark_name in pose_landmark_names:
        body_3d_xyz_header.append(f"{landmark_name}_x")
        body_3d_xyz_header.append(f"{landmark_name}_y")
        body_3d_xyz_header.append(f"{landmark_name}_z")

    right_hand_3d_xyz_header = []
    for landmark_name in hand_landmark_names:
        right_hand_3d_xyz_header.append(f"right_hand_{landmark_name}_x")
        right_hand_3d_xyz_header.append(f"right_hand_{landmark_name}_y")
        right_hand_3d_xyz_header.append(f"right_hand_{landmark_name}_z")

    left_hand_3d_xyz_header = []
    for landmark_name in hand_landmark_names:
        left_hand_3d_xyz_header.append(f"left_hand_{landmark_name}_x")
        left_hand_3d_xyz_header.append(f"left_hand_{landmark_name}_y")
        left_hand_3d_xyz_header.append(f"left_hand_{landmark_name}_z")

    face_3d_xyz_header = []
    for landmark_number in range(last_face_marker_index - first_face_marker_index):
        face_3d_xyz_header.append(f"face_{str(landmark_number).zfill(4)}_x")
        face_3d_xyz_header.append(f"face_{str(landmark_number).zfill(4)}_y")
        face_3d_xyz_header.append(f"face_{str(landmark_number).zfill(4)}_z")


    number_of_frames = mediapipe_3d_frame_trackedPoint_xyz.shape[0]
    body_flat = body_3d_xyz.reshape(number_of_frames, number_of_body_points*3)


    body_dataframe = pd.DataFrame(body_flat, columns=body_3d_xyz_header)
    body_dataframe.to_csv(str(mediapipe_3d_npy_path.parent / "mediapipe_body_3d_xyz.csv"), index=False)

    right_hand_flat = right_hand_3d_xyz.reshape(number_of_frames, number_of_hand_points*3)
    right_hand_dataframe = pd.DataFrame(right_hand_flat, columns=right_hand_3d_xyz_header)
    right_hand_dataframe.to_csv(str(mediapipe_3d_npy_path.parent / "mediapipe_right_hand_3d_xyz.csv"), index=False)

    left_hand_flat = left_hand_3d_xyz.reshape(number_of_frames, number_of_hand_points*3)
    left_hand_dataframe = pd.DataFrame(left_hand_flat, columns=left_hand_3d_xyz_header)
    left_hand_dataframe.to_csv(str(mediapipe_3d_npy_path.parent / "mediapipe_left_hand_3d_xyz.csv"), index=False)

    face_flat = face_3d_xyz.reshape(number_of_frames, number_of_face_points*3)
    face_dataframe = pd.DataFrame(face_flat, columns=face_3d_xyz_header)
    face_dataframe.to_csv(str(mediapipe_3d_npy_path.parent / "mediapipe_face_3d_xyz.csv"), index=False)

