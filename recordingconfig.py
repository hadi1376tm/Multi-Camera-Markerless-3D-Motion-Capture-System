

from pathlib import Path


dataFolder = '_Data'


folder_setup = ['RawVideos','SyncedVideos','CalVideos','DataArrays','DLCData','OpenPoseData','MediaPipeData','imOut']

default_parameters = {'path_to_save': str(Path.home()),'dlc_config_paths':[], 'rotations':{},'parameters':{'exposure':-5,'resWidth':640,'resHeight':480,'framerate':25,'codec':'DIVX'}, 'blenderEXEpath':None}
saved_parameters = default_parameters
parameters_for_yaml = {'default':default_parameters,'saved':saved_parameters}

