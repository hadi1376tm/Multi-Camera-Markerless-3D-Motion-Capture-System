import cv2


def createVideo(session):

    vidSavePath = str(session.sessionPath / "{}_outVid.mp4".format(session.sessionID))
    fps = 30
    shape = 1078, 647
    frame_array = []

    for filename in session.imOutPath.glob("*.png"):
        filename = str(filename)

        img = cv2.imread(filename)
        resized = cv2.resize(img, shape)
        frame_array.append(resized)
    out = cv2.VideoWriter(vidSavePath, cv2.VideoWriter_fourcc(*"DIVX"), fps, shape)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()


def createBodyTrackingVideos(session):

    vidSavePath = str(session.sessionPath / "{}_outVid.mp4".format(session.sessionID))


