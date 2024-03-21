from pathlib import Path
import os
from startup import startup
from aniposelib.boards import CharucoBoard
from aniposelib.boards import Checkerboard
import numpy as np
from scipy.signal import savgol_filter # for smoothing
import cv2
import time


import session,animation3D,recordingconfig,calibrate,\
    the_mediapipe,reconstruct3D,mediapipenpytocsv,mediapipe_angle, theGUI


def Main(
        sessionID=None, # Identifying string to use for this session.
        step=1,        # Which processing step to start from
        useMediaPipe=True, # Whether or not to use the MediaPipe tracking method
        runMediaPipe=True, # If False will use previously processed data
        mediapipe_model_complexity = 2,# 1 for lite MediaPipe pose model
        select_joints_angle = False, # Whether to select joints to calculate angles
        setDataPath = False, # Triggers the GUI for choosing data saving and loading path
        save_output3dVid = True, # whether to save the matplotlib 3D skeleton animation
        showAnimation = False, # whether to open a window showing 3D skeleton animation
        plotAxRange = 1500, # Range of 3D animation plot range in XYZ axis millimeters
        the3D_reconstructionConfidenceThreshold = .5, # Threshold 'confidence' value to include a point in the 3D reconstruction
        smoothing = True, # Do data outlier removal and smoothing
        outlier_threshold = 3, # Threshold of outlier removal filter
        filtering_window_size = 20, # Outlier removal sliding window size
        savgol_smoothWinLength = 15, # Savgol smoothing filter sliding window size, must be odd
        charucoSquareSize = 120,# Lenth of black square side of board in millimeters
        trim_cal_videos = True, # Create new calibration videos
        calVideoFrameLength = 0.5, # What portion of the videos to use in the calibration. -1 uses the whole recording
        animationStartFrame = 0, # From which frame of the video to start the animation
        use_saved_calibration = False, # whether to use a calibration file from a previous session
        calibration_board_type = "Checker" # or "Charuco"
        ):

    sessionObject = session.Session()

    sessionObject.sessionID = sessionID
    sessionObject.useMediaPipe = useMediaPipe
    sessionObject.setDataPath = setDataPath
    sessionObject.userDataPath = None
    sessionObject.dataFolderName = recordingconfig.dataFolder
    sessionObject.animationStartFrame = animationStartFrame
    sessionObject.get_synced_unix_timestamps = True
    sessionObject.use_saved_calibration = use_saved_calibration
    sessionObject.select_joints_angle = select_joints_angle
    sessionObject.selectedRightJoints = ['right_hip', 'right_elbow','right_knee'] # default plot angles
    sessionObject.selectedLeftJoints = ['left_hip', 'left_elbow','left_knee']

    sessionObject.selectedJoints = []
    smoothOrder = 3 # savgol_filter smooth order

    # %% Startup
    sessionObject._module_path = Path(__file__).parent


    startup.get_user_preferences(sessionObject,step)

    if step > 1:
        startup.get_data_folder_path(sessionObject)

        if sessionObject.sessionID == None:
            subfolders = [f.path for f in os.scandir(sessionObject.dataFolderPath) if f.is_dir()]  # copy-pasta from who knows where
            sessionObject.sessionID = Path(subfolders[-1]).stem  # grab the name of the last folder in the list of subfolders

        print('Running ' + str(sessionObject.sessionID) + ' from ' + str(sessionObject.dataFolderPath))


    if calibration_board_type == "Checker":
        board = Checkerboard(5, 4,
                            square_length = charucoSquareSize)
        sessionObject.board = board
    elif calibration_board_type == "Charuco":
        board = CharucoBoard(7, 5,
                            square_length = charucoSquareSize,#mm
                            marker_length = charucoSquareSize*.8,#mm
                            marker_bits=4, dict_size=250)
    # %% Initialization

    sessionObject.initialize(step)
    # saving time consumed for each step log
    with open(sessionObject.sessionPath / 'parameters and times.txt', 'w') as time_file:
        time_file.write("parameters: \n")
        for arg_name, arg_value in locals().items():
            if arg_name != 'file' and arg_name != 'sessionObject' and arg_name != 'time_file' and arg_name != 'board':
                time_file.write(f"{arg_name} = {arg_value}, ")
        time_file.write(" \n ------------------------------------------ \n\n")

    # triggering angle selection GUI
    if select_joints_angle:
        sessionObject.selectedJoints = theGUI.Select_joints_angle_calculator()
        sessionObject.selectedRightJoints = sessionObject.selectedJoints[0]
        sessionObject.selectedLeftJoints = sessionObject.selectedJoints[1]
        print("left side selected angles: " + str(sessionObject.selectedLeftJoints))
        print("right side selected angles: " + str(sessionObject.selectedRightJoints))
    else:
        print("default angles selected")
        print("left side selected angles: " + str(sessionObject.selectedLeftJoints))
        print("right side selected angles: " + str(sessionObject.selectedRightJoints))


    if step <= 2:
        start_time = time.time()
        print('-Step 3: Calibration')
        # Using Anipose to calculate calibration parameters each camera based on detected charuco boards.
        # This information is used to create a camera projection matrix for each camera, which is used in the 3d reconstruction'
        if sessionObject.numFrames is None:
            a_sync_vid_path = list(sessionObject.syncedVidPath.glob('*.mp4'))
            temp_cap = cv2.VideoCapture(str(a_sync_vid_path[0]))
            sessionObject.numFrames = temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            temp_cap.release()

        sessionObject.cgroup, sessionObject.mean_charuco_fr_mar_xyz = calibrate.CalibrateCaptureVolume(sessionObject,board,trim_cal_videos, calVideoFrameLength)
        print('-Anipose Calibration Was Successful')
        end_time = time.time() - start_time
        with open(sessionObject.sessionPath / 'parameters and times.txt', 'a') as time_file:
            time_file.write(f'spent time for -Stage 3: Calibration- : {end_time:.2f} seconds \n')
        print(f'spent time for -Stage 3: Calibration- : {end_time:.2f} seconds \n')
    else:
        print('-Skipping Calibration')


    # This step tracks the body in the 2d videos from videos to produce the data that will be combined with the camera projection matrix from the calibration step to produce 3D outout.
    if step <= 3:
        start_time = time.time()
        print('-Step 3: 2D Pose estimation')

        if sessionObject.useMediaPipe:

            print('-Running MediaPipe Pose')
            if runMediaPipe:
                if not mediapipe_model_complexity == 1: # too make sure (only 1 or 2 complexity in available in MediaPipe pose)
                    mediapipe_model_complexity = 2
                the_mediapipe.runMediaPipe(session=sessionObject,mediapipe_model_complexity=mediapipe_model_complexity)
                sessionObject.mediaPipeData_nCams_nFrames_nImgPts_XYC = the_mediapipe.parseMediaPipe(sessionObject,find_Face_mediapipe=False)

            else:
                print('-`runMediaPipe` set to False, loading MediaPipe data from npy file')
                sessionObject.mediaPipeData_nCams_nFrames_nImgPts_XYC = np.load(sessionObject.dataArrayPath/'mediaPipeData_2d.npy', allow_pickle=True)


            sessionObject.mediaPipeSkel_fr_mar_xyz, sessionObject.mediaPipeSkel_reprojErr = reconstruct3D.reconstruct3D(sessionObject,sessionObject.mediaPipeData_nCams_nFrames_nImgPts_XYC, confidenceThreshold=the3D_reconstructionConfidenceThreshold)

            smoothed_data_ArrayPath = sessionObject.dataArrayPath / 'CSVfiles'
            smoothed_data_ArrayPath.mkdir(exist_ok=True)

            np.save(sessionObject.dataArrayPath/'mediaPipeSkel_3d.npy', sessionObject.mediaPipeSkel_fr_mar_xyz) #save data to npy
            np.save(sessionObject.dataArrayPath/'mediaPipeSkel_reprojErr.npy', sessionObject.mediaPipeSkel_reprojErr) #save data to npy
            mediapipenpytocsv.mediapipetocsv(sessionObject.dataArrayPath / 'mediaPipeSkel_3d.npy')

            #outlier removal and smoothing
            if smoothing:
                print("-Smoothing 3D data")
                data = sessionObject.mediaPipeSkel_fr_mar_xyz
                filtered_data = np.zeros_like(data)
                # Loop through each window of the data and remove outliers
                # data points in the window that are bigger than (outlier_threshold * standard deviations) away from the mean
                # value will be considered outliers and replaced with the mean value.
                for i in range(data.shape[0] - filtering_window_size + 1):
                    for j in range(data.shape[1] - filtering_window_size + 1):
                        # Extract the window of data
                        window = data[i:i + filtering_window_size, j:j + filtering_window_size]

                        # Calculate the mean and standard deviation of the window
                        mean = np.mean(window)
                        std = np.std(window)

                        # Replace any outliers in the window with the mean value
                        window[np.abs(window - mean) > outlier_threshold * std] = mean

                        # Copy the filtered window to the output array
                        filtered_data[i:i + filtering_window_size, j:j + filtering_window_size] = window
                # applying savgol filter for smoothing data
                for dim in range(sessionObject.mediaPipeSkel_fr_mar_xyz.shape[2]):
                    for mm in range(sessionObject.mediaPipeSkel_fr_mar_xyz.shape[1]):
                        sessionObject.mediaPipeSkel_fr_mar_xyz[:, mm, dim] = savgol_filter(
                            filtered_data[:, mm, dim], savgol_smoothWinLength, smoothOrder)

            np.save(sessionObject.dataArrayPath/'mediaPipeSkel_3d_smoothed.npy', sessionObject.mediaPipeSkel_fr_mar_xyz)
            np.save(smoothed_data_ArrayPath/'mediaPipeSkel_3d_smoothed.npy', sessionObject.mediaPipeSkel_fr_mar_xyz)

            mediapipenpytocsv.mediapipetocsv(smoothed_data_ArrayPath / 'mediaPipeSkel_3d_smoothed.npy')

            mediapipe_angle.calculate_angles(selectedJoints=sessionObject.selectedJoints,csvpath= smoothed_data_ArrayPath)
            mediapipe_angle.calculate_distances(csvpath= smoothed_data_ArrayPath)

        sessionObject.save_session()


        sessionObject.save_session()
        sessionObject.syncedVidList = []
        sessionObject.save_session()
        end_time = time.time() - start_time
        with open(sessionObject.sessionPath / 'parameters and times.txt', 'a') as time_file:
            time_file.write(f'spent time for -Step 3: 2D Point Trackers- : {end_time:.2f} seconds \n')
        print(f'spent time for -Step 3: 2D pose estimation- : {end_time:.2f} seconds \n')
    else:

        print('-Skipping 2D point tracking')

    if step <= 4:
        start_time = time.time()
        print('-Step 4: Creating animation')

        animation3D.animationmaker(sessionObject, startFrame=sessionObject.animationStartFrame, azimuth=-90, elevation=-82,
                                   usingMediaPipe=useMediaPipe,
                                   save_output3dVid = save_output3dVid,
                                   showAnimation=showAnimation, plotAxRange= plotAxRange,
                                   selectedRightJoints=sessionObject.selectedRightJoints,
                                   selectedLeftJoints=sessionObject.selectedLeftJoints)

        end_time = time.time() - start_time
        with open(sessionObject.sessionPath / 'parameters and times.txt', 'a') as time_file:
            time_file.write(f'spent time for -Step 4: Creating animation- : {end_time:.2f} seconds \n')
        print(f'spent time for -Step 4: Creating animation- : {end_time:.2f} seconds \n')
    else:
        print('-Skipping 3D Animation')
    print('-Process finished!')
    print('-Session Data folder is at: ')
    print(str(sessionObject.sessionPath))

    print('------------------- FINISHED ------------------')

