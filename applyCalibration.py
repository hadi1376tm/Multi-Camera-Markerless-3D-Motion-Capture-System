import cv2

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


if __name__ == '__main__':
    input_vid_address = './shoes/1401-5-15/android/VID_20220806_200933.mp4'
    calib_yml_address = './shoes/1401-5-15/android/calibration_charuco.yml'
    cap = cv2.VideoCapture(input_vid_address)
    if not cap.isOpened():
        print('Cannot open video')
        # exit()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter('uni_result.avi',
                             cv2.VideoWriter_fourcc(*'XVID'),
                             fps, (frame_width, frame_height))
    mtx, dist = load_coefficients(calib_yml_address)

    cap = cv2.VideoCapture(input_vid_address)
    while cap.isOpened():
        success, image = cap.read()
        if (not success):
            print("cap read failed")
            # If loading a video, use 'break' instead of 'continue'.
            break

        dst = cv2.undistort(image, mtx, dist, None, mtx)
        cv2.imshow("result",dst)
        result.write(dst)
        if cv2.waitKey(1) == 27:
            continue
    result.release()
    print("done")