import os

from configs import config


def load_annotations(root, tracking_root):
    data = []

    for video_folder_name in os.listdir(root):
        current_video_folder_path = os.path.join(root, video_folder_name)

        if not os.path.isdir(current_video_folder_path):
            continue

        annotations_path = os.path.join(current_video_folder_path, "annotations.txt")

        if not os.path.isfile(annotations_path):
            continue

        video_id = int(video_folder_name)

        with open(annotations_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 2:
                    continue

                image_filename = parts[0]
                group_label = parts[1]

                clip_id = os.path.splitext(image_filename)[0]

                image_path = os.path.join(
                    current_video_folder_path,
                    clip_id,
                    image_filename
                )

                tracking_path = os.path.join(
                    tracking_root,
                    str(video_id),
                    clip_id,
                    f"{clip_id}.txt"
                )

                data.append({
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "image_path": image_path,
                    "tracking_path": tracking_path,
                    "group_label": group_label
                })

    return data


def split_dataset(data):
    train_data = []
    val_data = []
    test_data = []

    for sample in data:
        if sample["video_id"] in config.TRAIN_VIDEOS:
            train_data.append(sample)
        elif sample["video_id"] in config.VAL_VIDEOS:
            val_data.append(sample)
        else:
            test_data.append(sample)

    return train_data, val_data, test_data


def prepare_data():
    data = load_annotations(config.VIDEOS_ROOT, config.TRACKING_ROOT)
    train_data, val_data, test_data = split_dataset(data)
    return train_data, val_data, test_data
