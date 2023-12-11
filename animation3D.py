
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.interactive(False)

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.signal import savgol_filter
import moviepy.editor as mp

import os
import time
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from rich import pretty
pretty.install()  # makes all print more visualized
from rich.console import Console

console = Console()


#colors from Taylor Davis branding -
default_dark = np.array([37, 67, 66])/255
default_green = np.array([53, 93, 95])/255
default_blue = np.array([14, 90, 253])/255
default_red = np.array([217, 61, 67])/255
default_purple = np.array([128,0,128])/255
default_gray = np.array([128,128,128])/255

fps = 30


# main function of making 3D animation
def animationmaker(
    session=None,
    startFrame=1,           #  from what frame start making the animation
    azimuth=-90,
    elevation=-70,          # matplot parameters (POV)
    usingMediaPipe=False,     # which model is used
    save_output3dVid = True, # save mp4 3D animation file
    showAnimation =False,   # open 3D output window
    plotAxRange = 1300,      # range of 3D animation plot range in XYZ axis millimeters
    zero_center = False,    # plot center will be (0,0,0) else center will be average of subject positions
    selectedRightJoints = [],
    selectedLeftJoints = [],
):
    def update_figure(frameNum):

        if usingMediaPipe:
            mp_skel_dottos = matplot_artist_objects['mp_skel_dottos']
            mp_skel_trajectories = figure_data['mediaPipe_skel_trajectories|mar|fr_dim']


        # function to update the lines for each 3d body segment
        def update_3d_skeleton_segments(key, data_fr_mar_xyz):
            """
            updates the Artist of each body segment with the current frame data.
            """
            split = key.split('_')

            if split[0] == 'mp':
                dict_of_segments_idxs = dict_mediaPipe_SegmentIdx_dicts[key]
            try:
                dict_of_artists = matplot_artist_objects[key]
            except:
                print('Error:', key)

            for thisItem in dict_of_segments_idxs.items():
                segName = thisItem[0]
                segArtist = dict_of_artists[segName]
                segArtist.set_data((data_fr_mar_xyz[frameNum, dict_of_segments_idxs[segName], 0],
                                    data_fr_mar_xyz[frameNum, dict_of_segments_idxs[segName], 1]))
                segArtist.set_3d_properties(data_fr_mar_xyz[frameNum, dict_of_segments_idxs[segName], 2])


        if usingMediaPipe:
            update_3d_skeleton_segments('mp_body', mediaPipe_skel_xyz)
            update_3d_skeleton_segments('mp_rHand', mediaPipe_skel_xyz)
            update_3d_skeleton_segments('mp_lHand', mediaPipe_skel_xyz)


        # mediapipe data
        if usingMediaPipe:
            marNum = -1
            for thisSkelDotto, thisTraj in zip(mp_skel_dottos, mp_skel_trajectories):
                marNum += 1
                thisSkelDotto.set_data(thisTraj[frameNum, 0:2])
                thisSkelDotto.set_3d_properties(thisTraj[frameNum, 2])


        # function to update the lines for each body segment
        def update_2d_segments_func(vidNum, key):
            """
            updates the Artist of each body segment with the current frame data.
            """
            split = key.split('_')

            if split[0] == 'mp':
                dict_of_segments_idxs = dict_mediaPipe_SegmentIdx_dicts[key]
                dict_of_artists = thisVidMediaPipeArtist_dict[key]

            for thisItem in dict_of_segments_idxs.items():
                segName = thisItem[0]
                segArtist = dict_of_artists[segName]


                if split[0] == 'mp':
                    xData = mediaPipe_nCams_nFrames_nImgPts_XYC[vidNum, frameNum, dict_of_segments_idxs[segName], 0]
                    yData = mediaPipe_nCams_nFrames_nImgPts_XYC[vidNum, frameNum, dict_of_segments_idxs[segName], 1]

                xDataMasked = np.ma.masked_array(xData, mask=(xData == 0))
                yDataMasked = np.ma.masked_array(yData, mask=(xData == 0))
                segArtist.set_data(xDataMasked, yDataMasked)

        # update video frames
        for vidNum, thisVidArtist in enumerate(vidAristList):
            vidCap_Objs_List[vidNum].set(cv2.CAP_PROP_POS_FRAMES, frameNum)
            success, image = vidCap_Objs_List[vidNum].read()

            if success:
                thisVidArtist.set_array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                vidCap_Objs_List[vidNum].set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, image = vidCap_Objs_List[vidNum].read()


            if usingMediaPipe:
                thisVidMediaPipeArtist_dict = list_of_vid_MediaPipe_Artist_dicts[vidNum]
                update_2d_segments_func(vidNum, 'mp_body')
                update_2d_segments_func(vidNum, 'mp_rHand')
                update_2d_segments_func(vidNum, 'mp_lHand')

        if usingMediaPipe:
            rightSideTimeSeriesAx.set_xlim([(frameNum / fps) - timeRange, (frameNum / fps) + timeRange])
            leftSideTimeSeriesAx.set_xlim([(frameNum / fps) - timeRange, (frameNum / fps) + timeRange])
            for thisArtistKey in rightSideCurrTimeArtists:

                rightSideCurrTimeArtists[thisArtistKey][0].set_xdata(frameNum / fps)
                if not thisArtistKey == 'blackLine':
                    rightSideCurrTimeArtists[thisArtistKey][0].set_ydata(rightSideCurrTimedata[thisArtistKey][frameNum])

            for thisArtistKey in leftSideCurrTimeArtists:

                leftSideCurrTimeArtists[thisArtistKey][0].set_xdata(frameNum / fps)
                if not thisArtistKey == 'blackLine':
                    leftSideCurrTimeArtists[thisArtistKey][0].set_ydata(leftSideCurrTimedata[thisArtistKey][frameNum])

        framenumber_text.set_text("Frame#: " + str(frameNum))



    # create figure
    fig = plt.figure(dpi=200)
    # fig2 = plt.figure(dpi=200)
    plt.ion()


    # create 3D axis
    ax3d = fig.add_subplot(projection='3d')
    # ax3d_2 = fig2.add_subplot(projection='3d')
    # ax3d_2.set_position([0.10, .20, .8, .8])  # [left, bottom, width, height])

    ax3d.set_position([-.15, .25, .8, .8])  # [left, bottom, width, height])


    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.tick_params(labelsize=4)

    # ax3d_2.set_xlabel('X')
    # ax3d_2.set_ylabel('Y')
    # ax3d_2.set_zlabel('Z')
    # ax3d_2.tick_params(labelsize=4)


    if usingMediaPipe:
        try:
            mediaPipe_skel_xyz = np.load(session.dataArrayPath / 'mediaPipeSkel_3d_smoothed.npy')
            mediaPipe_nCams_nFrames_nImgPts_XYC = np.load(session.dataArrayPath / 'mediaPipeData_2d.npy')
        except:
            print('No mediaPipe data found.')


    charuco_fr_mar_xyz = None
    figure_data = dict()

    if usingMediaPipe:
        mediaPipe_trajectories = [mediaPipe_skel_xyz[:, markerNum, :] for markerNum in
                                  range(mediaPipe_skel_xyz.shape[1])]
        figure_data['mediaPipe_skel_trajectories|mar|fr_dim'] = [mediaPipe_skel_xyz[:, markerNum, :] for
                                                                 markerNum in range(mediaPipe_skel_xyz.shape[1])]
        figure_data['mediaPipe_skel_xyz'] = mediaPipe_skel_xyz

        dict_mediaPipe_SegmentIdx_dicts, dict_of_mp_skel_lineColor = formatMediaPipeStickIndices()  # these will help us draw body and hands stick figures

    def build_3d_segment_artist_dict(data_fr_mar_xyz,
                                     dict_of_list_of_segment_idxs,
                                     segColor='k',
                                     lineWidth=1,
                                     lineStyle='-',
                                     markerType=None,
                                     marSize=12,
                                     markerEdgeColor='k', ):
        """
        Builds a dictionary of line artists for each 3D body segment.
        """
        segNames = list(dict_of_list_of_segment_idxs)

        dict_of_artist_objects = dict()
        for segNum, segName in enumerate(segNames):

            # determine color of segment, based on class of 'segColor' input
            if isinstance(segColor, str):
                theRGBA = segColor
            elif isinstance(segColor, np.ndarray):
                theRGBA = segColor
            elif isinstance(segColor, dict):
                theRGBA = segColor[segName]
            elif isinstance(segColor, list):
                try:
                    theRGBA = segColor[segNum]
                except:
                    print('Not enough colors provided, using Black instead')
                    theRGBA = 'k'
            else:
                theRGBA = 'k'

            if isinstance(segName, str):
                idxsOG = dict_of_list_of_segment_idxs[segName]
            else:
                idxsOG

            if isinstance(idxsOG, int) or isinstance(idxsOG, float):
                idxs = [idxsOG]
            elif isinstance(idxsOG, dict):
                idxs = idxsOG[0]
            else:
                idxs = idxsOG.copy()

            dict_of_artist_objects[segName] = ax3d.plot(
                data_fr_mar_xyz[startFrame, idxs, 0],
                data_fr_mar_xyz[startFrame, idxs, 1],
                data_fr_mar_xyz[startFrame, idxs, 2],
                linestyle=lineStyle,
                linewidth=lineWidth,
                markerSize=marSize,
                marker=markerType,
                color=theRGBA,
                markeredgecolor=markerEdgeColor,
            )[0]
        return dict_of_artist_objects

    matplot_artist_objects = dict()


    if usingMediaPipe:
        matplot_artist_objects['mp_body'] = build_3d_segment_artist_dict(mediaPipe_skel_xyz,
                                                                         dict_mediaPipe_SegmentIdx_dicts['mp_body'],
                                                                         segColor=dict_of_mp_skel_lineColor)
        matplot_artist_objects['mp_rHand'] = build_3d_segment_artist_dict(mediaPipe_skel_xyz,
                                                                          dict_mediaPipe_SegmentIdx_dicts['mp_rHand'],
                                                                          segColor=np.append(default_red, 1),
                                                                          markerType='.', markerEdgeColor=default_red,
                                                                          lineWidth=1, marSize=2)
        matplot_artist_objects['mp_lHand'] = build_3d_segment_artist_dict(mediaPipe_skel_xyz,
                                                                          dict_mediaPipe_SegmentIdx_dicts['mp_lHand'],
                                                                          segColor=np.append(default_blue, 1),
                                                                          markerType='.', markerEdgeColor=default_blue,
                                                                          lineWidth=1, marSize=2)


    if usingMediaPipe:
        matplot_artist_objects['mp_skel_dottos'] = [
            ax3d.plot(thisTraj[0, 0:1], thisTraj[1, 0:1], thisTraj[2, 0:1], 'm.', markersize=1)[0] for thisTraj in
            mediaPipe_trajectories]


    # find mean center

    if usingMediaPipe:
        numFrames = mediaPipe_skel_xyz.shape[0]
        mx = np.nanmean(mediaPipe_skel_xyz[int(numFrames / 2), :, 0])
        my = np.nanmean(mediaPipe_skel_xyz[int(numFrames / 2), :, 1])
        mz = np.nanmean(mediaPipe_skel_xyz[int(numFrames / 2), :, 2])
    #center will be average of subject positions
    if np.isnan(mx) or np.isnan(my) or np.isnan(mz):
        mx = 0
        my = 0
        mz = 0

    axRange = plotAxRange
    #axRange = session.board.square_length * 10

    if zero_center:
        mx = 0
        my = 0
        mz = 0
    # Setting the axes properties
    ax3d.set_xlim3d([mx - axRange, mx + axRange])
    ax3d.set_ylim3d([my - axRange*0.80, my + axRange*0.80])
    ax3d.set_zlim3d([mz - axRange, mz + axRange])

    ax3d.view_init(azim=azimuth, elev=elevation)

    ## make video axes
    syncedVidPathListAll = list(sorted(session.syncedVidPath.glob('*.mp4')))
    numVids = len(syncedVidPathListAll)

    syncedVidPathList = syncedVidPathListAll.copy()

    vidAxesList = []
    vidAristList = []
    vidCap_Objs_List = []

    list_of_vid_MediaPipe_Artist_dicts = []

    def build_2d_segment_artist_dict(vidNum, data_nCams_nFrames_nImgPts_XYC, dict_of_list_of_segment_idxs, segColor='k',lineWidth=1, lineStyle='-'):
        """
        Builds a dictionary of line artists for each body 2d segment.
        """
        segNames = list(dict_of_list_of_segment_idxs)

        dict_of_artist_objects = dict()
        for segNum, segName in enumerate(segNames):

            # determine color of segment, based on class of 'segColor' input
            if isinstance(segColor, str):
                theRGBA = segColor
            elif isinstance(segColor, np.ndarray):
                theRGBA = segColor
            elif isinstance(segColor, dict):
                theRGBA = segColor[segName].copy()
                theRGBA[-1] = .75
            elif isinstance(segColor, list):
                try:
                    theRGBA = segColor[segNum]
                except:
                    print('Not enough colors provided, using Black instead')
                    theRGBA = 'k'
            else:
                theRGBA = 'k'

            xData = data_nCams_nFrames_nImgPts_XYC[vidNum, startFrame, dict_of_list_of_segment_idxs[segName], 0]
            yData = data_nCams_nFrames_nImgPts_XYC[vidNum, startFrame, dict_of_list_of_segment_idxs[segName], 1]

            xDataMasked = np.ma.masked_where(xData, np.isnan(xData))
            yDataMasked = np.ma.masked_where(yData, np.isnan(yData))

            dict_of_artist_objects[segName] = thisVidAxis.plot(
                xDataMasked,
                yDataMasked,
                linestyle=lineStyle,
                linewidth=lineWidth,
                color=theRGBA,
            )[0]
        return dict_of_artist_objects

    if numVids < 4:
        numRows = numVids
        numCols = 1
    else:
        numRows = int(np.ceil(numVids / 2))
        numCols = 2

    vidAxGridSpec = fig.add_gridspec(numRows, numCols, left=.65, bottom=.025, right=.95, top=.9, wspace=.05, hspace=.05)

    for thisVidNum, thisVidPath in enumerate(syncedVidPathList):
        # make subplot for figure

        if numVids < 4:
            thisVidAxis = fig.add_subplot(vidAxGridSpec[thisVidNum])
        elif numVids % 2 > 0:  # if odd number videos plot the first vid across 2 spots
            if thisVidNum == 0:
                thisVidAxis = fig.add_subplot(vidAxGridSpec[0, :])
            else:
                thisVidAxis = fig.add_subplot(vidAxGridSpec[thisVidNum + 1])
        else:
            thisVidAxis = fig.add_subplot(vidAxGridSpec[thisVidNum])

        thisVidAxis.set_axis_off()

        vidAxesList.append(thisVidAxis)

        # create video capture object
        thisVidCap = cv2.VideoCapture(str(thisVidPath))

        # create artist object for each video
        success, image = thisVidCap.read()

        assert success == True, "{} - failed to load the image".format(thisVidPath.stem)  # make sure we have a frame

        vidAristList.append(thisVidAxis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        vidCap_Objs_List.append(thisVidCap)


        if usingMediaPipe:
            vidMediaPipeArtist_dict = dict()
            vidMediaPipeArtist_dict['mp_body'] = build_2d_segment_artist_dict(thisVidNum,
                                                                              mediaPipe_nCams_nFrames_nImgPts_XYC,
                                                                              dict_mediaPipe_SegmentIdx_dicts[
                                                                                  'mp_body'], segColor='g')
            vidMediaPipeArtist_dict['mp_rHand'] = build_2d_segment_artist_dict(thisVidNum,
                                                                               mediaPipe_nCams_nFrames_nImgPts_XYC,
                                                                               dict_mediaPipe_SegmentIdx_dicts[
                                                                                   'mp_rHand'],
                                                                               segColor=np.append(default_red, .75),
                                                                               lineWidth=.5)
            vidMediaPipeArtist_dict['mp_lHand'] = build_2d_segment_artist_dict(thisVidNum,
                                                                               mediaPipe_nCams_nFrames_nImgPts_XYC,
                                                                               dict_mediaPipe_SegmentIdx_dicts[
                                                                                   'mp_lHand'],
                                                                               segColor=np.append(default_blue, .75),
                                                                               lineWidth=.5)
            list_of_vid_MediaPipe_Artist_dicts.append(vidMediaPipeArtist_dict)


    if usingMediaPipe:
        fps = 30
        timestamps = np.arange(0, numFrames) / fps
        df = pd.read_csv(session.dataArrayPath / 'CSVfiles' / 'mediapipe_angles.csv')
        colors = [default_blue,default_red,default_green,default_purple,default_gray]
        left_joints = selectedLeftJoints
        left_joints.sort()
        right_joints = selectedRightJoints
        right_joints.sort()
        # get angles
        left_angles = df[left_joints]
        right_angles = df[right_joints]

        linePlotWidth = .45
        linePlotHeight = .14

        leftSideTimeSeriesAx = fig.add_subplot(position=[.07, .225, linePlotWidth, linePlotHeight])
        leftSideTimeSeriesAx.set_title('Angles', fontsize=7, pad=2)

        # leftSideTimeSeriesAx.plot(timestamps, left_elbow, color=default_red, linewidth=.75, label='left elbow')
        # leftSideTimeSeriesAx.plot(timestamps, left_shoulder, color=default_blue, linewidth=.75, label='left shoulder')

        for i, joint in enumerate(left_joints):
            leftSideTimeSeriesAx.plot(timestamps, left_angles[joint], color=colors[i], linewidth=.75, label=joint)

        timeRange = 4
        anglelimRange = 180

        #left side joints plot:
        leftSideCurrTimeArtists = dict()
        leftSideCurrTimedata = dict()
        leftSideCurrTimeArtists['blackLine'] = leftSideTimeSeriesAx.plot([startFrame / fps, startFrame / fps],
                                                           [0, anglelimRange * 2], color='k', linewidth=1)

        for i, joint in enumerate(left_joints):
            leftSideCurrTimeArtists[joint] = leftSideTimeSeriesAx.plot([startFrame / fps], [left_angles[joint][startFrame]],
                                               markeredgecolor=colors[i], markerfacecolor='k', marker='o', markersize=3)
            leftSideCurrTimedata[joint] = left_angles[joint]


        leftSideTimeSeriesAx.tick_params(labelsize=6, direction='in', width=.5)
        leftSideTimeSeriesAx.tick_params(pad=2)

        leftSideTimeSeriesAx.set_ylabel('Left Angles', fontsize=7, labelpad=3)
        leftSideTimeSeriesAx.set_ylim([0, anglelimRange])
        leftSideTimeSeriesAx.set_xlim([(startFrame / fps) - timeRange, (startFrame / fps) + timeRange])

        for axis in ['top', 'bottom', 'left', 'right']:
            leftSideTimeSeriesAx.spines[axis].set_linewidth(0.5)

        leftSideTimeSeriesAx.legend(loc='upper left', fontsize=4)
        # right side joints plot:

        rightSideTimeSeriesAx = fig.add_subplot(position=[.07, 0.05, linePlotWidth, linePlotHeight])

        for i, joint in enumerate(right_joints):
            rightSideTimeSeriesAx.plot(timestamps, right_angles[joint], color=colors[i], linewidth=.75, label=joint)

        rightSideCurrTimeArtists = dict()
        rightSideCurrTimedata = dict()
        rightSideCurrTimeArtists['blackLine'] = rightSideTimeSeriesAx.plot([startFrame / fps, startFrame / fps], [0, anglelimRange],
                                                           color='k', linewidth=1)

        for i, joint in enumerate(right_joints):
            rightSideCurrTimeArtists[joint] = rightSideTimeSeriesAx.plot([startFrame / fps], [right_angles[joint][startFrame]],
                                               markeredgecolor=colors[i], markerfacecolor='k', marker='o', markersize=3)
            rightSideCurrTimedata[joint] = right_angles[joint]


        rightSideTimeSeriesAx.tick_params(labelsize=6, direction='in', width=.2)
        rightSideTimeSeriesAx.tick_params(pad=2)

        rightSideTimeSeriesAx.set_ylabel('Right Angles', fontsize=7, labelpad=0)
        rightSideTimeSeriesAx.set_xlabel('Time(sec)', fontsize=8, labelpad=0)

        rightSideTimeSeriesAx.set_ylim([0, anglelimRange])

        rightSideTimeSeriesAx.set_xlim([(startFrame / fps) - timeRange, (startFrame / fps) + timeRange])

        for axis in ['top', 'bottom', 'left', 'right']:
            rightSideTimeSeriesAx.spines[axis].set_linewidth(0.5)
        rightSideTimeSeriesAx.legend(loc='upper left', fontsize=4)

    # both side plots created
    ####
    framenumber_text = fig.text(.5, .95, "Frame#: " + str(startFrame), fontsize=10, horizontalalignment='center')
    # framenumber_text2 = fig2.text(.5, .95, "Frame#: " + str(startFrame), fontsize=10, horizontalalignment='center')

    # Creating the Animation object
    out_animation = animation.FuncAnimation(fig, update_figure, range(startFrame, numFrames), fargs=(),
                                            interval=1, blit=False)
    # out_animation2 = animation.FuncAnimation(fig2, update_figure, range(startFrame, numFrames), fargs=(),
    #                                         interval=1, blit=False)
    if save_output3dVid:
        gifSavePath = '{}_3DAnimationOutput_Vid.gif'.format(str(session.sessionPath / session.sessionID))
        # gifSavePath2 = '{}_animVid2.gif'.format(str(session.sessionPath / session.sessionID))

        with console.status('Saving animation for {}'.format(session.sessionID)):
            fps = 30
            tik = time.time()
            Writer = animation.writers['pillow']
            writer = Writer(fps=fps)
            out_animation.save(gifSavePath, writer=writer)
            gif_filepath = mp.VideoFileClip(gifSavePath)
            gif_filepath.write_videofile(gifSavePath.replace('.gif', '.mp4'))
            os.remove(gifSavePath)
            # print("creating only 3dpose animation video")
            # Writer2 = animation.writers['pillow']
            # writer2 = Writer2(fps=fps)
            # out_animation2.save(gifSavePath2, writer=writer)
            # gif_filepath2 = mp.VideoFileClip(gifSavePath2)
            # gif_filepath2.write_videofile(gifSavePath2.replace('.gif', '.mp4'))
            # os.remove(gifSavePath2)

            # tok = time.time() - tik
            # print(f'Took {tok} seconds to save the animation')

    try:
        if showAnimation:
            with console.status('Playing Skeleton animation! Close the `matplotlib` window to continue...'):
                plt.pause(0.01)
                plt.draw()
    except:
        pass



def formatMediaPipeStickIndices():
    """
    generate dictionary of arrays, each containing the 'connect-the-dots' order to draw a given body segment

    returns:
    mediaPipeBody_Segment_Ids= a dictionary of arrays containing indices of individual body segments.
    mediaPipeHandIds = a dictionary of arrays containing indices of individual hand segments.
    dict_of_mp_skel_lineColor = a dictionary of arrays, each containing the color to use for a the body segment
    """
    dict_mediaPipe_SegmentIdx_dicts = dict()

    # make body dictionary
    mediaPipeBody_Segment_Ids = dict()
    mediaPipeBody_Segment_Ids['mp_head'] = [8, 6, 5, 4, 0, 10, 9, 0, 1, 2, 3, 7]
    mediaPipeBody_Segment_Ids['mp_tors'] = [12, 11, 24, 23, 12]
    mediaPipeBody_Segment_Ids['mp_rArm'] = [12, 14, 16, 18, 20, 16, 22]
    mediaPipeBody_Segment_Ids['mp_lArm'] = [11, 13, 15, 17, 19, 15, 21]
    mediaPipeBody_Segment_Ids['mp_rLeg'] = [24, 26, 28, 30, 32, 28, ]
    mediaPipeBody_Segment_Ids['mp_lLeg'] = [23, 25, 27, 29, 31, 27, ]
    dict_mediaPipe_SegmentIdx_dicts['mp_body'] = mediaPipeBody_Segment_Ids

    # make colors dictionary
    # mediaPipeBodyColor = np.array([180, 50, 0]) / 255
    # mediaPipeRightColor = np.array([240, 10, 120]) / 255
    # mediaPipeLeftColor = np.array([30, 110, 170]) / 255
    # mediaPipeBodyColor = np.array([130, 90, 0]) / 255
    mediaPipeBodyColor = np.array([0, 153, 51]) / 255
    mediaPipeRightColor = np.array([220, 10, 160]) / 255
    mediaPipeLeftColor = np.array([10, 170, 170]) / 255
    dict_of_mp_skel_lineColor = dict()

    dict_of_mp_skel_lineColor['mp_head'] = np.append(mediaPipeBodyColor, .5)
    dict_of_mp_skel_lineColor['mp_tors'] = np.append(mediaPipeBodyColor, 1)
    dict_of_mp_skel_lineColor['mp_rArm'] = np.append(mediaPipeRightColor, 1)
    dict_of_mp_skel_lineColor['mp_lArm'] = np.append(mediaPipeLeftColor, 1)
    dict_of_mp_skel_lineColor['mp_rLeg'] = np.append(mediaPipeRightColor, 1)
    dict_of_mp_skel_lineColor['mp_lLeg'] = np.append(mediaPipeLeftColor, 1)

    # hand maps
    mediaPipeHandIds = dict()
    rHandIDstart = 33
    lHandIDstart = rHandIDstart + 21

    mediaPipeHandIds['mp_thumb'] = np.array([0, 1, 2, 3, 4, ])
    mediaPipeHandIds['mp_index'] = np.array([0, 5, 6, 7, 8, ])
    mediaPipeHandIds['mp_bird'] = np.array([0, 9, 10, 11, 12, ])
    mediaPipeHandIds['mp_ring'] = np.array([0, 13, 14, 15, 16, ])
    mediaPipeHandIds['mp_pinky'] = np.array([0, 17, 18, 19, 20, ])

    rHand_dict = copy.deepcopy(
        mediaPipeHandIds.copy())  # copy.deepcopy() is necessary to make sure the dicts are independent of each other
    lHand_dict = copy.deepcopy(rHand_dict)

    for key in rHand_dict:
        rHand_dict[key] += rHandIDstart
        lHand_dict[key] += lHandIDstart

    dict_mediaPipe_SegmentIdx_dicts['mp_rHand'] = rHand_dict
    dict_mediaPipe_SegmentIdx_dicts['mp_lHand'] = lHand_dict

    return dict_mediaPipe_SegmentIdx_dicts, dict_of_mp_skel_lineColor


if __name__ == '__main__':
    animationmaker()
