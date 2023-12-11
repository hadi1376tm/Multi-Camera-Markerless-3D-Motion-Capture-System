# Multi-camera-Markerless-3D-Motion-capture-System


This repository contains a Python project designed for Multi-camera Markerless 3D Motion Capture System.

## Setup and Execution

### Prerequisites

- Python 3.7
- Installation of required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Program

1. **Clone the repository** to your local machine.

2. **Configure `start.py`**:
    - Set the main function parameters in `start.py` based on your requirements. Example:
    ```python
    main.Main(
         sessionID="test",
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
        calVideoFrameLength = 0.5,
            # What portion of the videos to use in the calibration. -1 uses the whole recording
        animationStartFrame = 0,
            # from which frame of the video to start the animation
        use_saved_calibration = False,
            # whether to use a calibration file from a previous session
    )
    ```

3. **Directory Setup**:
    - Create a directory named `_Data` in this format: `_Data/[your session name]`.
    - Inside `_Data/[your session name]`, create folders named `CalVideos` and `SyncedVideos`.
    - Place synchronized calibration videos with charuco board (Acessable in "charuco_board" folder in the repository) in the `CalVideos` folder and corresponding subject videos, with the same names in the `SyncedVideos` folder.

4. **Example Data**:
    - You can download and use the example (also contaning the final outputs) data from [this link](https://drive.google.com/drive/folders/1zCEKPteKGi976wzmMp3WFPF8V88fDnbS?usp=drive_link) and extract it to your system.

    - Execute `start.py` with the provided parameters:
    ```bash
    main.Main(step=2,use_saved_calibration=False,setDataPath = True,sessionID="eceL1",charucoSquareSize=60,
          useMediaPipe=True, runMediaPipe=True,
          the3D_reconstructionConfidenceThreshold= 0.3,
          mediapipe_model_complexity = 2,select_joints_angle=False,
          showAnimation=False, plotAxRange=2000, animationStartFrame=0,
          trim_cal_videos=False,smoothing=False)
    ```
    - You will be shown a window to provide the path to the downloaded _Data folder. Set the path and click on "Proceed".
    - The program will start proccesing the videos and provide the final output, including a 3D animation video, 3D coordinates, and angle data saved as CSV files.

## Notes

- Please note that the program's execution time can be significant and may vary based on your system hardware. It could take several minutes to generate the outputs

