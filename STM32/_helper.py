import os

# Define the path to the directory containing folders
directory_path = '/Users/TomasPacheco/Documents/MA2/MLuC/Project/ML_Model/fruits/fruits-360/Training'

# Get a list of all folders in the directory
folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]

# Iterate through each folder
for folder in folders:
    # Construct the current and new paths
    current_path = os.path.join(directory_path, folder)
    new_folder_name = folder.replace(" ", "_")  # Replace whitespace with underscores (or any other character)
    new_path = os.path.join(directory_path, new_folder_name)

    # Rename the folder
    os.rename(current_path, new_path)

    print(f"Renamed '{folder}' to '{new_folder_name}'")