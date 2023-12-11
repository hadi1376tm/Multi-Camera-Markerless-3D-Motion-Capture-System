# Multi-camera-Markerless-3D-Motion-capture-System


This repository contains a Python project designed for [describe project purpose]. Ensure you have Python 3.7 installed and install the required packages listed in the `requirements.txt` file before running the program.

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
        sessionID="eceL1",
        step=2,
        use_saved_calibration=False,
        setDataPath=True,
        # Add other parameters as needed
    )
    ```

3. **Directory Setup**:
    - Create a directory named `_Data` in this format: `_Data/[your session name]`.
    - Inside `_Data`, create folders named `CalVideos` and `SyncedVideos`.
    - Place synchronized calibration videos in the `CalVideos` folder and corresponding subject videos in the `SyncedVideos` folder.

4. **Download Example Data**:
    - Download the example data from [this link](https://drive.google.com/drive/folders/1zCEKPteKGi976wzmMp3WFPF8V88fDnbS?usp=drive_link) and extract it to your system.

5. **Run the Program**:
    - Execute `start.py` with the provided parameters:
    ```bash
    python start.py
    ```
6. **Follow On-Screen Prompts**:
    - When prompted, select the path to the downloaded `_Data` folder and continue the program.

## Notes

- The program generates various outputs including 3D animation videos, 3D coordinates, and angle data saved as CSV files.
- Note that the program execution time can be substantial, depending on your system hardware.

