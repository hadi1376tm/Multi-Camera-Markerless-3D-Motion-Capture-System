# Given intrinsic parameters
intrinsic_params = {
    'azure_kinect_0_color': (976.7838346046938, 976.6954700661433, 1000.7015978283123, 784.6995695911885, 1536, 2048, 13.366022046162477, -11.15344330351859, 0.00016147564080253588, -0.008163149588165207, -88.58085389918612, 13.318338772065024, -12.813268592939618, -83.7145525534381),
    'azure_kinect_1_color': (972.7423807781718, 972.3850924769915, 1001.0666252429908, 784.8871379179792, 1536, 2048, -23.06510632923673, 223.65759622734413, 0.001576580975294305, -0.00849949885809845, 373.3356298674388, -23.140398630220908, 225.67456465888569, 351.94902429601433),
    'azure_kinect_2_color': (972.9417347642176, 972.8237671437809, 1011.2252926504023, 774.2363388087227, 1536, 2048, -5.722914191975826, 23.51554724143492, -0.0003031508530753926, -0.003248534044295182, -13.546530380262688, -5.78639265785433, 23.76926358551252, -14.441875071724118),
    'kinect_v2_1_color': (1051.813621943641, 1050.2902776992407, 547.3272604766133, 962.9661100011954, 1920, 1080, 2.870747955523379, 128.69386546899707, -0.00269840586891665, 0.01075096838028446, -37.70568439804439, 2.899566709579175, 128.32862017073415, -38.892762624365446),
    'kinect_v2_2_color': (1040.256926237827, 1039.9560342296354, 497.06098569517025, 993.3129441002038, 1920, 1080, 10.622736609531803, 197.54632596719804, 0.0017736238939001084, -0.007176410705924418, -556.6356298683487, 10.631081680905009, 196.45065045452043, -556.1522575764117),
    'event_camera': (1679.2962596007244, 1679.3779810251924, 401.29677177872213, 641.793926378788, 1280, 800, 164.78601546569507, 246.10433207416435, 0.002504744137648569, 0.002417041551607122, -101.92097783039958, 164.8400884185635, 280.735792757698, -96.63869752874388),
    'polar': (1867.0668669033637, 1860.8227843759773, 492.400259947796, 553.3939994915685, 1024, 1224, -0.24136709983251609, -108.27219894712556, -0.0008323616136531622, -0.011773896066462213, 192.91854316659033, -0.07941036588715922, -109.9566838660344, 195.77215236390904)
}

# Given extrinsic parameters
extrinsic_params = {
    'azure_kinect_2-kinect_v2_2': ([[0.36945632, 0.18551353, -0.91054201], [-0.17976142, 0.97562867, 0.12583529], [0.91169505, 0.11718969, 0.39380036]], [1959.11888943, -578.50765759, 1110.62345042]),
    'azure_kinect_0-azure_kinect_2': ([[0.5825479, 0.03953389, -0.81183436], [-0.14209833, 0.98838756, -0.05383399], [0.80027871, 0.14672118, 0.58140079]], [1634.30464635, 13.60120773, 951.56086836]),
    'polar-azure_kinect_0': ([[0.89448237, -0.0878222, 0.43839314], [0.07437111, 0.99608444, 0.04779883], [-0.44087438, -0.01015143, 0.89751141]], [-760.37471501, 165.61321671, 2638.39316552]),
    'event_camera-azure_kinect_0': ([[0.9540899, 0.08732972, -0.28650651], [-0.04307238, 0.98661213, 0.15729359], [0.2964072, -0.13773171, 0.94507817]], [647.48687916, -13.73014195, 1449.14857397]),
    'azure_kinect_0-azure_kinect_1': ([[0.66563952, -0.02430931, 0.7458774], [0.19384399, 0.97079569, -0.14135144], [-0.72065841, 0.23867295, 0.65091219]], [-1743.63137122, 38.39562383, 922.28690343]),
    'azure_kinect_1-kinect_v2_1': ([[0.44373607, 0.01910836, 0.89595378], [-0.14862876, 0.98749581, 0.05255017], [-0.88374645, -0.15648291, 0.44102756]], [-1720.2012324, -416.66131741, 1208.72464332])
}

# Function to format camera parameters

# Function to convert intrinsic parameters to the desired format
def intrinsic_to_text(name, params):
    size = [params[4], params[5]]
    matrix = f"[ [{params[0]}, 0.0, {params[2]}], [0.0, {params[1]}, {params[3]}], [0.0, 0.0, 1.0] ]"
    distortions = f"[ {', '.join(str(x) for x in params[6:10])} ]"
    rotation = f"[ {', '.join(str(x) for x in params[10:13])} ]"
    translation = "[ 0.0, 0.0, 0.0 ]"
    fisheye = "true"

    return (
        f"[cam_{name}]\n"
        f"name = \"{name}\"\n"
        f"size = {size}\n"
        f"matrix = {matrix}\n"
        f"distortions = {distortions}\n"
        f"rotation = {rotation}\n"
        f"translation = {translation}\n"
        f"fisheye = {fisheye}\n\n"
    )


# Function to convert extrinsic parameters to the desired format
def extrinsic_to_text(cam_pair, params):
    rotation = f"[ {', '.join(str(x) for x in params[0][0])}, {', '.join(str(x) for x in params[0][1])}, {', '.join(str(x) for x in params[0][2])} ]"
    translation = f"[ {', '.join(str(x) for x in params[1])} ]"

    return (
        f"[cam_{cam_pair}]\n"
        f"name = \"{cam_pair}\"\n"
        f"size = []\n"
        f"matrix = []\n"
        f"distortions = []\n"
        f"rotation = {rotation}\n"
        f"translation = {translation}\n"
        f"fisheye = false\n\n"
    )


# Create text for intrinsic parameters
intrinsic_text = ''
for idx, (cam_name, params) in enumerate(intrinsic_params.items()):
    intrinsic_text += intrinsic_to_text(idx + 1, params)

# Create text for extrinsic parameters
extrinsic_text = ''
for cam_pair, params in extrinsic_params.items():
    extrinsic_text += extrinsic_to_text(cam_pair, params)

# Combine both intrinsic and extrinsic texts
final_text = intrinsic_text + extrinsic_text

# Save the final text to a file
output_file_path = 'camera_parameters.txt'
with open(output_file_path, 'w') as output_file:
    output_file.write(final_text)

print(f"Camera parameters saved to '{output_file_path}'")