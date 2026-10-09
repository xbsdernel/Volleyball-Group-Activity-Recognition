# Dataset root directory
ROOT_DIR = "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos"

# Video split indices (used to partition parsed annotations into
# train / val / test sets)
TRAIN_VIDEOS = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38,
                39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_VIDEOS = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_VIDEOS = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

# Training hyperparameters
LR = 1e-3
BATCH_SIZE = 64  # Per-GPU batch size
EPOCHS = 50

# Image preprocessing
IMAGE_SIZE = (128, 128)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Model
NUM_CLASSES = 8

# Early stopping
PATIENCE = 5

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "best_model.pth"

# DataLoader settings
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Performance Optimization Flags
USE_AMP = True             # Automatic Mixed Precision (FP16)
USE_CHANNELS_LAST = True    # Channels Last memory format (NHWC)
CUDNN_BENCHMARK = True     # cuDNN autotuner for fixed shape (128x128)
USE_DDP = True             # DistributedDataParallel for multi-GPU
SEED = 42                  # Reproducibility seed

