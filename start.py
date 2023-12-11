"""
Set the Main() parameters down here and run this file (start.py)
These are the parameters to set and their default values:
"""

import main

main.Main(step=4,use_saved_calibration=True,setDataPath = False,sessionID="eceL1",charucoSquareSize=60,
          useMediaPipe=True, runMediaPipe=True,
          the3D_reconstructionConfidenceThreshold= 0.3,
          mediapipe_model_complexity = 1,select_joints_angle=False,
          showAnimation=False, plotAxRange=2000, animationStartFrame=0,
          trim_cal_videos=False,smoothing=False)

"""
---->>> def Main(
        sessionID=None,
            # Identifying string to use for this session.
        step=1,
            # Which processing step to start from
        useMediaPipe=True,
            # Whether or not to use the MediaPipe tracking method
        runMediaPipe=True,
            # If False will use previously processed data
        mediapipe_model_complexity = 2,
            # 1 for lite MediaPipe pose model
        select_joints_angle = False,
            # Whether to select  joints to calculate angles
        setDataPath = False,
            # Triggers the GUI for choosing data saving and loading path
        save_output3dVid = True,
            # whether to save the matplotlib 3D skeleton animation
        showAnimation = False,
            # whether to open a window showing 3D skeleton animation
        plotAxRange = 1500,
            # Range of 3D animation plot range in XYZ axis millimeters
        the3D_reconstructionConfidenceThreshold = .5,
            # Threshold 'confidence' value to include a point in the 3D reconstruction
        smoothing = True,
            # Do data outlier removal and smoothing
        outlier_threshold = 3,
            # Threshold of outlier removal filter
        filtering_window_size=20,
            # Outlier removal sliding window size
        savgol_smoothWinLength = 15,
            # Savgol smoothing filter sliding window size, must be odd
        charucoSquareSize = 120,
            # Length of black square side of board in millimeters
        trim_cal_videos = True,
            # Create new calibration videos
        calVideoFrameLength = 0.5,
            # What portion of the videos to use in the calibration. -1 uses the whole recording
        animationStartFrame = 0,
            # from which frame of the video to start the animation
        use_saved_calibration = False,
            # whether to use a calibration file from a previous session
        )
"""



