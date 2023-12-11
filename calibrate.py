import pickle
import glob 

from aniposelib.cameras import CameraGroup
import numpy as np
import cv2


import reconstruct3D, anipose
from rich.progress import track
from rich import print
from rich.console import Console

from scipy.spatial.transform import Rotation

console = Console()

def pin_camera_zero_to_origin(_anipose_camera_group_object):
    original_translation_vectors = _anipose_camera_group_object.get_translations()
    camera_0_translation = original_translation_vectors[0, :]
    altered_translation_vectors = np.zeros(original_translation_vectors.shape)
    for this_camera_number in range(original_translation_vectors.shape[0]):
        altered_translation_vectors[this_camera_number, :] = original_translation_vectors[this_camera_number,
                                                            :] - camera_0_translation

    _anipose_camera_group_object.set_translations(altered_translation_vectors)
    print(f"original translation vectors: {original_translation_vectors}")
    print(f"altered translation vectors: {_anipose_camera_group_object.get_translations()}")
    return _anipose_camera_group_object

def rotate_cameras_so_camera_zero_aligns_with_XYZ(_anipose_camera_group_object):
    original_rotations_euler = _anipose_camera_group_object.get_rotations()
    original_translation_vectors =  _anipose_camera_group_object.get_translations()
    camera_rotation_matrix_list = [Rotation.from_euler('xyz',original_rotations_euler[this_cam_num,:]).as_matrix()  for this_cam_num in range(original_rotations_euler.shape[0])]
    
    rotated_translation_vectors = [camera_rotation_matrix_list[0] @ this_tx for this_tx in original_translation_vectors]
    _anipose_camera_group_object.set_rotations(rotated_translation_vectors)
    return _anipose_camera_group_object

def CalibrateCaptureVolume(session,board,trim_cal_videos, calVideoFrameLength = 1):
    """ 
    Check if a previous calibration yaml exists, and if not, create a set of shortened calibration videos and run Anipose functions
    to create a calibration yaml. Takes the 2D charuco board points and reconstructs them into 3D points that are saved out
    into the DataArrays
    """
    
    session.calVidPath.mkdir(exist_ok = True)
    session.dataArrayPath.mkdir(exist_ok = True)
    calibrationVideoPath = session.calVidPath


    
    if session.use_saved_calibration:
        saved_calibration_folder = session._module_path/'calibration'
        saved_calibration_file_path = saved_calibration_folder/'previous_calibration.toml'


        cam_names = [i+1 for i in range(session.numCams)]
        cgroup = anipose.CameraGroup.from_names( cam_names, fisheye=True)


        cgroup = CameraGroup.load(saved_calibration_file_path)
        charuco_nCams_nFrames_nImgPts_XY = np.load(saved_calibration_folder/'charuco_2d_points.npy')

        session.cameraCalFilePath = session.sessionPath /"{}_calibration.toml".format(session.sessionID)
        cgroup.dump(session.cameraCalFilePath) 

        session.cgroup = cgroup 

        camera_calibration_info_dict = cgroup.get_dicts()
        camera_calibration_pickle_path = session.sessionPath / "{}_calibration.pickle".format(session.sessionID)

        with open(str(camera_calibration_pickle_path), 'wb') as pickle_file:
            pickle.dump(camera_calibration_info_dict, pickle_file)
        
    else:

        if type(calVideoFrameLength)==int or type(calVideoFrameLength)==float:
            if calVideoFrameLength < 0: # if '-1' use the whole video
                cal_vid_frame_range = [0, session.numFrames]
                calibrationVideoPath = session.calVidPath
            else:
                if calVideoFrameLength>0 and calVideoFrameLength<=1: #if between 0 and 1, use as a percentage of the total video length
                    cal_vid_frame_range = [0, round(calVideoFrameLength * session.numFrames)]
                else: #otherwise, just use the input value as the number of frames to use        
                    cal_vid_frame_range = [0, calVideoFrameLength]
        elif type(calVideoFrameLength)==list:
            if len(calVideoFrameLength) ==2:
                cal_vid_frame_range = calVideoFrameLength

        if trim_cal_videos:
            create_CalibrationVideos(session, cal_vid_frame_range)

        vidnames = []
        cam_names = []

        for count, thisVidPath in enumerate(calibrationVideoPath.glob("*.mp4"), start=1):
            vidnames.append([str(thisVidPath)])
            cam_names.append(str(count))
            session.numCams = count

        
        cgroup = anipose.CameraGroup.from_names( cam_names, fisheye=True)  # Looking through their code... it looks lke the 'fisheye=True' doesn't do much (see 2020-03-29 obsidian note)

        calibrationFile = "{}_calibration.toml".format(session.sessionID)

        session.cameraCalFilePath = session.sessionPath / calibrationFile

        error,charuco_data, charuco_frames = cgroup.calibrate_videos(vidnames, board)
    
        cgroup = pin_camera_zero_to_origin(cgroup)
        cgroup.dump(session.cameraCalFilePath)

        camera_calibration_info_dict = cgroup.get_dicts()
        camera_calibration_pickle_path = session.sessionPath / "{}_calibration.pickle".format(session.sessionID)
        
        
        with open(str(camera_calibration_pickle_path), 'wb') as pickle_file:
            pickle.dump(camera_calibration_info_dict, pickle_file)



        session.cgroup = cgroup
        n_frames = cal_vid_frame_range[1]-cal_vid_frame_range[0]
        startframe = 0
        n_trackedPoints = 24

        charuco_nCams_nFrames_nImgPts_XY = np.empty([session.numCams, n_frames, n_trackedPoints,  2])
        charuco_nCams_nFrames_nImgPts_XY[:] = np.nan

        for cam in range(session.numCams):
            for charCount, thisCharFrame in enumerate(charuco_frames):
                try:
                    charuco_nCams_nFrames_nImgPts_XY[cam, thisCharFrame, :,:] = np.squeeze(charuco_data[charCount][cam]["filled"])
                except:
                    # print("failed frame:", frame)
                    continue
                
        saved_calibration_folder = session._module_path/'calibration'
        saved_calibration_folder.mkdir(exist_ok=True, parents=True)
        calibration_toml_path = saved_calibration_folder/'previous_calibration.toml'
        calibration_charuco2d_npy_path = saved_calibration_folder/'charuco_2d_points.npy'
        cgroup.dump(calibration_toml_path)
        np.save(calibration_charuco2d_npy_path,charuco_nCams_nFrames_nImgPts_XY)
        
        
    charuco2d_filename = session.dataArrayPath/'charuco_2d_points.npy'
    np.save(charuco2d_filename,charuco_nCams_nFrames_nImgPts_XY)



    session.charuco_nCams_nFrames_nImgPts_XY = charuco_nCams_nFrames_nImgPts_XY

    charuco_fr_mar_xyz, charuco_reprojErr= reconstruct3D.reconstruct3D(
        session, charuco_nCams_nFrames_nImgPts_XY
    )

    mean_charuco_fr_mar_xyz = np.nanmean(charuco_fr_mar_xyz, axis=0)


    np.save(session.dataArrayPath/'charuco_3d_points.npy', charuco_fr_mar_xyz)
    np.save(session.dataArrayPath/'charuco_3d_reprojErr.npy', charuco_reprojErr)

    return cgroup, charuco_fr_mar_xyz


def create_CalibrationVideos(session, calVideoFrameLength):
    """ 
    Based on the desired length of the calibration videos (for the anipose functions), create new videos trimmed 
    to that specific length
    """  
    vidList =  glob.glob(str(session.syncedVidPath) + "/*.mp4")
    if len(calVideoFrameLength)==1:
        framelist = list(range(calVideoFrameLength))
    elif len(calVideoFrameLength)==2:
        framelist = list(range(calVideoFrameLength[0], calVideoFrameLength[1]))
    else:
        Exception('calVideoFrameLength must be either 1 or 2 elements long')
    
    codec = "DIVX"
    for count, vid in enumerate(vidList, start=1):
        cam_name = "Cam{}".format(count)
        cap = cv2.VideoCapture(vid)
        fourcc = cv2.VideoWriter_fourcc(*codec)

        # grab resolution parameters from the videos
        resWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        framerate = int(cap.get(cv2.CAP_PROP_FPS))

        saveName = (
            session.sessionID + "_trimmed_" + cam_name + ".mp4"
        )  # create a name for the trimmed video
        saveCalVidPath = str(
            session.calVidPath / saveName
        )  # create an output path for the function

        success, image = cap.read()
        out = cv2.VideoWriter(saveCalVidPath, fourcc, framerate, (resWidth, resHeight))
        print("Trimming " + cam_name + " to frames {}-{} for Anipose Calibration".format(framelist[0], framelist[-1]))
        for frame in track(framelist):
            cap.set(
                cv2.CAP_PROP_POS_FRAMES, frame
            )  # set the video to the frame that we need
            success, image = cap.read()
            out.write(image)
        cap.release()
        out.release()
