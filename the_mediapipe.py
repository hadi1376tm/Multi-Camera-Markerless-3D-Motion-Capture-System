
import numpy as np
from rich.progress import Progress
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

mediapipe_indices = [
    'nose',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_ear',
    'right_ear',
    'mouth_left',
    'mouth_right',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot_index',
    'right_foot_index'
    ]


def annotate_image_with_mediapipe_data(image, mediapipe_results):
    mp_drawing.draw_landmarks(image=image,
                              landmark_list=mediapipe_results.face_landmarks,
                              connections=mp_holistic.FACEMESH_CONTOURS,
                              landmark_drawing_spec=None,
                              connection_drawing_spec=mp_drawing_styles
                              .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(image=image,
                              landmark_list=mediapipe_results.face_landmarks,
                              connections=mp_holistic.FACEMESH_CONTOURS,
                              landmark_drawing_spec=None,
                              connection_drawing_spec=mp_drawing_styles
                              .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(image=image,
                              landmark_list=mediapipe_results.pose_landmarks,
                              connections=mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles
                              .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=mediapipe_results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
            .get_default_hand_connections_style())
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=mediapipe_results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
            .get_default_hand_connections_style())
    return image


def runMediaPipe(session,mediapipe_model_complexity):
    """ 
    Run MediaPipe on synced videos, and save body tracking data to be parsed 
    """  

    mediapipe_video_save_path = session.sessionPath/'mediapipe_output_Videos'
    mediapipe_video_save_path.mkdir(exist_ok=True)


    with mp_holistic.Holistic(
                            static_image_mode = False,
                            model_complexity = mediapipe_model_complexity, # = 1 for faster run
                            smooth_landmarks = True,
                            enable_segmentation = True,
                            smooth_segmentation = True
                            ) as holistic:

        eachCamerasData = []  # Create an empty list that holds each cameras data
        eachCameraResolution = {'Height':[],'Width':[]}
        vid_count = 0
        for (thisVidPath) in (session.syncedVidPath.iterdir()):  # Run MediaPipe 'Holistic' (body, hands, face) tracker on each video in the raw video folder
            if ( thisVidPath.suffix.lower() == ".mp4" ):  # NOTE - at some point we should build some list of 'synced video names' and check against that
                vid_count += 1

                name_for_saved_video = session.sessionID + '_mediapipe_output_video_{}'.format(str(vid_count)) + '.mp4'

                mediaPipe_dataList = []  # Create an empty list for mediapipes data
               
                cap = cv2.VideoCapture(str(thisVidPath))
                frameNum = -1
                numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                framerate = cap.get(cv2.CAP_PROP_FPS)

                writer = cv2.VideoWriter(str(mediapipe_video_save_path / name_for_saved_video),
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         framerate,
                                         (video_frame_width, video_frame_height))
                success, image = cap.read()  # load first image from video
                # print("cap success")
                with Progress() as progress:
                    progressBar = progress.add_task(
                        "[magenta]MediaPiping: {}".format(thisVidPath.name),
                        total=numFrames,
                    )  # make progressbar

                    while success:

                        if frameNum % 5 == 0:
                            progress.update(progressBar, advance=5)

                        frameNum += 1

                        image_height, image_width, _ = image.shape
                        
                        # Mediapipe main process
                        results = holistic.process(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        )
                        annotated_image = annotate_image_with_mediapipe_data(image, results)

                        writer.write(annotated_image)
                        mediaPipe_dataList.append(
                            results
                        )  # Append data to mediapipe data list
                        writer.write(annotated_image)
                        success, image = cap.read()  # load next image from video
                        cv2.putText(image, "frame#: " + str(frameNum), (50, 50),
                                    cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 0, 0), 3)
                    writer.release()
                    cap.release()
                eachCameraResolution["Height"].append(image_height)
                eachCameraResolution["Width"].append(image_width)

                eachCamerasData.append(
                    mediaPipe_dataList
                )  # Append that cameras data for every frame to the camera datalist

    session.image_height = image_height
    session.image_width = image_width
    session.mediaPipeData = eachCamerasData
    session.eachCameraResolution = eachCameraResolution



def parseMediaPipe(session,find_Face_mediapipe):
    """ 
    Parse through saved MediaPipe data, and save out a numpy array of 2D points
    """  
    numCams = len(session.mediaPipeData)  # Get number of cameras
    numFrames = len(session.mediaPipeData[0])  # Get number of frames
    numBodyPoints = 33
    numFacePoints = 468
    numHandPoints = 21

    numTrackedPoints = (
        numBodyPoints + numHandPoints * 2 + numFacePoints
    )  # Get total points

    # Create  array of nans the size of number of cams, frame, points, XYC
    mediaPipeData_nCams_nFrames_nImgPts_XYC = np.empty(
        (int(numCams), int(numFrames), int(numTrackedPoints), 3)
    )  # create empty array
    mediaPipeData_nCams_nFrames_nImgPts_XYC[:] = np.NaN  # Fill it with NaNs!

    for camNum in range(numCams):  # Loop through each camera
        for frNum in range(numFrames):  # Loop through each frame
            
            thisFrame_X_body = np.empty(numBodyPoints)
            thisFrame_X_body[:] = np.nan
            thisFrame_Y_body = thisFrame_X_body.copy()
            thisFrame_C_body = thisFrame_X_body.copy()

            thisFrame_X_face = np.empty(numFacePoints)
            thisFrame_X_face[:] = np.nan
            thisFrame_Y_face = thisFrame_X_face.copy()
            thisFrame_C_face = thisFrame_X_face.copy()

            thisFrame_hands = np.empty(numHandPoints)
            thisFrame_hands[:]= np.nan

            thisFrame_X_lefthand = thisFrame_hands.copy()
            thisFrame_Y_lefthand = thisFrame_hands.copy()
            thisFrame_C_lefthand = thisFrame_hands.copy()

            thisFrame_X_righthand = thisFrame_hands.copy()
            thisFrame_Y_righthand = thisFrame_hands.copy()
            thisFrame_C_righthand = thisFrame_hands.copy()

            full_landmarks = True

            try:
                # pull out ThisFrame's mediapipe data
                thisFrame_poseDataLandMarks = session.mediaPipeData[camNum][frNum].pose_landmarks.landmark  # body ('pose') data
                # stuff body data into pre-allocated nan array
                thisFrame_X_body[:numBodyPoints] = [pp.x for pp in thisFrame_poseDataLandMarks]  # PoseX data - Normalized screen coords (in range [0, 1]) - need multiply by image resultion for pixels
                thisFrame_Y_body[:numBodyPoints] = [pp.y for pp in thisFrame_poseDataLandMarks]  # PoseY data
                thisFrame_C_body[:numBodyPoints] = [pp.visibility for pp in thisFrame_poseDataLandMarks]  #'visibility' is MediaPose's 'confidence' measure in range [0,1]
            except:
                full_landmarks = False

            # Right hand data
            try:
                thisFrame_rHandDataLandMarks = session.mediaPipeData[camNum][frNum].right_hand_landmarks.landmark  # right hand data
                thisFrame_X_righthand[:numHandPoints] = [pp.x for pp in thisFrame_rHandDataLandMarks]  # PoseX data - Normalized screen coords (in range [0, 1]) - need multiply by image resultion for pixels
                thisFrame_Y_righthand[:numHandPoints] = [pp.y for pp in thisFrame_rHandDataLandMarks]  # PoseY data
                thisFrame_C_righthand[:numHandPoints] = [pp.visibility for pp in thisFrame_rHandDataLandMarks]  #'visibility' is MediaPose's 'confidence' measure in range [0,1]
            except:
                full_landmarks = False

            # Left hand data
            try:
                thisFrame_lHandDataLandMarks = session.mediaPipeData[camNum][frNum].left_hand_landmarks.landmark  # left hand data
                thisFrame_X_lefthand[:numHandPoints ] = [pp.x for pp in thisFrame_lHandDataLandMarks]  
                thisFrame_Y_lefthand[:numHandPoints] = [pp.y for pp in thisFrame_lHandDataLandMarks]
                thisFrame_C_lefthand[:numHandPoints] = [pp.visibility for pp in thisFrame_lHandDataLandMarks] 
            except:
                full_landmarks = False

            # FaceMeshData
            if find_Face_mediapipe:
                try:
                    thisFrame_faceDataLandMarks = session.mediaPipeData[camNum][frNum].face_landmarks.landmark  # face mesh data
                    thisFrame_X_face[:numFacePoints] = [pp.x for pp in thisFrame_faceDataLandMarks]
                    thisFrame_Y_face[:numFacePoints] = [pp.y for pp in thisFrame_faceDataLandMarks]
                    thisFrame_C_face[:numFacePoints] = [pp.visibility for pp in thisFrame_faceDataLandMarks]
                except:
                    full_landmarks = False


            thisFrame_X = np.concatenate((thisFrame_X_body,thisFrame_X_righthand,thisFrame_X_lefthand,thisFrame_X_face))
            thisFrame_Y = np.concatenate((thisFrame_Y_body,thisFrame_Y_righthand,thisFrame_Y_lefthand,thisFrame_Y_face))
            thisFrame_C = np.concatenate((thisFrame_C_body,thisFrame_C_righthand,thisFrame_C_lefthand,thisFrame_C_face))
            # put frame's data into mediaPipeData array
            mediaPipeData_nCams_nFrames_nImgPts_XYC[camNum, frNum, :, 0] = thisFrame_X
            mediaPipeData_nCams_nFrames_nImgPts_XYC[camNum, frNum, :, 1] = thisFrame_Y
            mediaPipeData_nCams_nFrames_nImgPts_XYC[camNum, frNum, :, 2] = thisFrame_C


    # convert from normalized screen coordinates to pixel coordinates
    for camera in range(numCams):
        mediaPipeData_nCams_nFrames_nImgPts_XYC[camera, :, :, 0] *= session.eachCameraResolution['Width'][camera] 
        mediaPipeData_nCams_nFrames_nImgPts_XYC[camera, :, :, 1] *= session.eachCameraResolution['Height'][camera] 

    # mediaPipeData_nCams_nFrames_nImgPts_XYC[:, :, 34:, 2] = 1
    # TODO

    np.save(session.dataArrayPath / "mediaPipeData_2d.npy", mediaPipeData_nCams_nFrames_nImgPts_XYC,)

    return mediaPipeData_nCams_nFrames_nImgPts_XYC
