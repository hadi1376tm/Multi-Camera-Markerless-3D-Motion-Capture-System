# Define the original camera parameters
cameras = [
    {
        "name": "1",
        "size": [1920, 1080],
        "matrix": [[1399.19, 0.0, 935.415], [0.0, 1396.03, 558.167], [0.0, 0.0, 1.0]],
        "distortions": [-0.291094, 0.195087, -0.000537235, 0.000431581],
        "rotation": [3.3465067176181433, -38.45216831179945, -5.4930963480915524],
        "translation": [12.28427843, 139.5977267, 264.2416486],
        "fisheye": True
    },
    {
        "name": "2",
        "size": [1920, 1080],
        "matrix": [[1392.84, 0.0, 915.859], [0.0, 1389.77, 563.308], [0.0, 0.0, 1.0]],
        "distortions": [-0.282817, 0.172464, 4.77793e-05, 0.000517998],
        "rotation": [-51.4127900050525, -38.6198702286429, 57.29537322878903],
        "translation": [24.01186267, 93.80917146, 365.1883193],  # Flattened translation for cam_1
        "fisheye": True
    },
    {
        "name": "3",
        "size": [1920, 1080],
        "matrix": [[1397.03, 0.0, 933.75], [0.0, 1393.45, 562.785], [0.0, 0.0, 1.0]],
        "distortions": [-0.287327, 0.184675, -0.000166486, 4.51601e-05],
        "rotation": [-121.72098848411586, -57.89372999741109, 120.56010184823437],
        "translation": [-7.904783205, 94.58720377, 339.7106049],
        "fisheye": True
    }
]

# Extract the translation of cam_0 without modifying it
cam_0_translation = cameras[0]["translation"][:]

# Apply translation adjustments for all cameras except cam_0
for camera in cameras:
    for i in range(3):
        camera["translation"][i] -= cam_0_translation[i]

# Print the updated camera parameters
for camera in cameras:
    print(f"Camera {camera['name']} - Updated Translation: {camera['translation']}")