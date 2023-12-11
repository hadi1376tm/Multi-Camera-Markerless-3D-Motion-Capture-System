import pickle

# Specify the path to your .pkl file
file_path = 'kinect_extrinsic_param.pkl'

# Read the .pkl file
with open(file_path, 'rb') as file:
    # Load the object from the file
    loaded_object = pickle.load(file)

# Create a new text file to save the loaded data
output_file_path = 'kinect_extrinsic_param.txt'

# Convert loaded_object to a string representation and write it to the text file
with open(output_file_path, 'w') as output_file:
    # Convert loaded_object to a string (you may need to adjust this based on your object)
    data_string = str(loaded_object)

    # Write the string representation to the text file
    output_file.write(data_string)

print(f"Data loaded from .pkl file and saved to '{output_file_path}' as text.")
