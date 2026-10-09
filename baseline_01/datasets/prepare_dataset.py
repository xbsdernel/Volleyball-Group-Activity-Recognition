"""
Parses the raw annotation files into train/val/test splits.
"""
import os

from configs import config


def load_annotations(root):
    """
    Walk through the dataset root directory and parse annotations.txt
    files. Returns a list of [video_folder(int), image_path, activity].
    """
    data = []
    if not os.path.exists(root):
        return data

    for video_folder in os.listdir(root):  # loop through video folders (0, 1, 2, ...)
        current_folder = os.path.join(root, video_folder)
        if not os.path.isdir(current_folder):
            continue

        annotations_path = os.path.join(current_folder, "annotations.txt")
        if not os.path.isfile(annotations_path):
            continue

        with open(annotations_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                image, activity = parts[0], parts[1]
                name, extension = os.path.splitext(os.path.basename(image))
                image_path = os.path.join(
                    current_folder,
                    name,
                    os.path.basename(image)
                )
                try:
                    data.append([int(video_folder), image_path, activity])
                except ValueError:
                    continue

    return data



def split_dataset(data):
    """
    Split parsed data into train/val/test sets based on the video
    indices defined in the config file.
    """
    train_data = []
    val_data = []
    test_data = []

    for line in data:
        for num in config.TRAIN_VIDEOS:
            if line[0] == num:
                train_data.append([line[1], line[2]])

    for line in data:
        for num in config.VAL_VIDEOS:
            if line[0] == num:
                val_data.append([line[1], line[2]])

    for line in data:
        for num in config.TEST_VIDEOS:
            if line[0] == num:
                test_data.append([line[1], line[2]])

    return train_data, val_data, test_data


def prepare_data():
    """
    Full data preparation pipeline: load annotations and split into
    train/val/test sets.
    """
    data = load_annotations(config.ROOT_DIR)
    train_data, val_data, test_data = split_dataset(data)
    return train_data, val_data, test_data
