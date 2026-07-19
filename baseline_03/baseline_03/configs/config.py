DATASET_ROOT = "/kaggle/input/datasets/ahmedmohamed365/volleyball"
VIDEOS_ROOT = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos"
TRACKING_ROOT = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation"

TRAIN_VIDEOS = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38,
                39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_VIDEOS = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_VIDEOS = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

LABEL_MAP = {
    "l_pass": 0,
    "r_pass": 1,
    "l_spike": 2,
    "r_spike": 3,
    "l_set": 4,
    "r_set": 5,
    "l_winpoint": 6,
    "r_winpoint": 7,
    "l-pass": 0,
    "r-pass": 1,
    "l-spike": 2,
    "r-spike": 3,
    "l-set": 4,
    "r-set": 5,
    "l-winpoint": 6,
    "r-winpoint": 7
}

LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 40

IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NUM_PLAYERS = 12
NUM_CLASSES = 8

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "final_model.pth"
