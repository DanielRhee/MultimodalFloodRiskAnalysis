import torch

ALIGNED_DATA_PATH = '../aligned_data.npz'
FLOOD_RISK_PATH = '../CreateDataset/flood_risk_dataset.npz'
LANDCOVER_MODEL_PATH = '../LandClassification/checkpoints/best_model.pth'

TILE_SIZE = 128
STRIDE = 64
MIN_VALID_RATIO = 0.7

TRAIN_VAL_SPLIT = 0.85
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_CHANNELS = 4
OUTPUT_CLASSES = 1

CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

RGB_MEAN = [104.0, 114.0, 99.0]
RGB_STD = [52.0, 46.0, 44.0]
