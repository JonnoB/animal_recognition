import os
import pathlib
import cv2
import pandas as pd
import torch

def list_files_recursively(folder_path):
    """
    Recursively lists all files in a given folder and its subfolders.
    Returns a list of absolute file paths.

    Args:
    folder_path (str): The path to the folder.

    Returns:
    List[str]: A list of absolute file paths.
    """
    files_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            files_list.append(os.path.abspath(os.path.join(root, file)))
    return files_list


def process_batch(batch, image_folder, model):
    """
    Processes a batch of images using the Megadetector YOLOv5 model to detect bounding boxes of animals.

    This function iterates over a batch of image data, loads each image, and applies the YOLOv5 model to detect animals.
    It outputs a DataFrame containing the class code, class name, file name, and bounding box coordinates for each detected animal.
    Batch processing is used to enhance the speed of image processing.

    Args:
        batch (pandas.DataFrame): A batch of data with each row containing image information.
            Expected columns are 'image_path_rel' for the relative image path,
            'class' for the class ID of the detected object, and 'type' for the object's type name.
        image_folder (str): The base directory path where images are stored.
        model: The YOLOv5 model object used for detecting objects in images.

    Returns:
        pandas.DataFrame: A DataFrame containing detection results with columns for class code, class name,
        file name, and bounding box coordinates (x, y, width, height). Returns None if no detections are made in the batch.
    """
    batch_results = []

    for _, row in batch.iterrows():
        file = row['image_path_rel']
        class_id = row['class']
        type_name = row['type']
        #file needs to be opened by openCV for some unknown and annoying reason, this definately slows the process down.
        #But I get a file not found error when trying to use yolo or PIL.
        # Load the image using OpenCV
        image_path = os.path.join(image_folder, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with your model
        full_res = model(image)
        results = full_res.pandas().xywhn[0]

        if not results.empty:
            results['class'] = class_id
            results['name'] = type_name
            results['file_name'] = file
            batch_results.append(results)

    print('batch complete returning batch')
    if batch_results:
        return pd.concat(batch_results, ignore_index=True)
    else:
        return None
    
    
def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data.iloc[i:i + batch_size]
        

def find_files_recursively(target_path):
    #returns a list of all files within a folder structure
    file_list = []
    for path, subdirs, files in os.walk(target_path):
        for name in files:
           file_list = file_list + [pathlib.PurePath(path, name)]
    return file_list

  
def populate_with_symlinks(file_list, folder_path):
    links_created = 0

    # Convert folder_path to an absolute path if it's not already
    folder_path = os.path.abspath(folder_path)

    # Ensure the destination directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for file_name in file_list:
        # Convert target_file to an absolute path
        target_file = os.path.abspath(file_name)
        destination_file = os.path.join(folder_path, os.path.basename(target_file))

        # Check if the source file exists
        if os.path.exists(target_file):
            # Remove existing symbolic link if it exists
            if os.path.exists(destination_file) or os.path.islink(destination_file):
                os.remove(destination_file)

            # Create a symbolic link using absolute paths
            os.symlink(target_file, destination_file)
            links_created += 1

    print(f'Total symbolic links created: {links_created}')