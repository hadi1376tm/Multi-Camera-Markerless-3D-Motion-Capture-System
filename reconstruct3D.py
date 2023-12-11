import anipose
import numpy as np
from rich.console import Console
console=Console()

def reconstruct3D(session, data_nCams_nFrames_nImgPts_XYC, confidenceThreshold=0.3):
    """
    Take a specifically formatted data array, and based on the camera calibration yaml, reconstruct a 3D image
    """

    if (
        session.cgroup is None
    ):  # load the calibration settings in to the session class
        calibrationFile = "{}_calibration.toml".format(session.sessionID)
        session.cameraConfigFilePath = session.sessionPath / calibrationFile
        session.cgroup = anipose.CameraGroup.load(session.cameraConfigFilePath)

    nCams, nFrames, nImgPts, nDims = data_nCams_nFrames_nImgPts_XYC.shape

    if nDims == 3:
        for camNum in range(nCams):
                          
            thisCamX = data_nCams_nFrames_nImgPts_XYC[camNum, :, :,0 ]
            thisCamY = data_nCams_nFrames_nImgPts_XYC[camNum, :, :,1 ]
            thisCamConf = data_nCams_nFrames_nImgPts_XYC[camNum, :, :, 2]
            # Remove low confidence landmarks
            thisCamX[thisCamConf < confidenceThreshold] = np.nan
            thisCamY[thisCamConf < confidenceThreshold] = np.nan



    if nDims == 2:
        data_nCams_nFrames_nImgPts_XY = data_nCams_nFrames_nImgPts_XYC
    elif nDims == 3:
        data_nCams_nFrames_nImgPts_XY = np.squeeze(data_nCams_nFrames_nImgPts_XYC[:, :, :, 0:2])

    Flatdata_nCams_nTotal_Points_XY = data_nCams_nFrames_nImgPts_XY.reshape(nCams, -1, 2)  # reshape data to collapse across 'frames' so it becomes [numCams, numFrames*numPoints, XY]

    console.print('Reconstructing 3d points.\n')
    data3d_flat = session.cgroup.triangulate(Flatdata_nCams_nTotal_Points_XY, progress=True)

    dataReprojerr_flat = session.cgroup.reprojection_error(data3d_flat, Flatdata_nCams_nTotal_Points_XY, mean=True)

    data_fr_mar_xyz = data3d_flat.reshape(nFrames, nImgPts, 3)
    dataReprojError = dataReprojerr_flat.reshape(nFrames, nImgPts)

    return data_fr_mar_xyz, dataReprojError
