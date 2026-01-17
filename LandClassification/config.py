import torch

DATASET_ROOT = "../FLAIR-HUB_Toy"
IMAGE_DIR_PATTERN = "*_AERIAL_RGBI"
LABEL_DIR_PATTERN = "*_AERIAL_LABEL-COSIA"

IMAGE_SIZE = 512
CROP_SIZE = 256
IN_CHANNELS = 4
NUM_CLASSES = 19

NORMALIZATION_MEAN = [104.0, 114.0, 99.0, 112.0]
NORMALIZATION_STD = [52.0, 46.0, 44.0, 49.0]

CLASS_NAMES = [
    "Building",
    "Greenhouse",
    "Swimming Pool",
    "Impervious Surface",
    "Pervious Surface",
    "Bare Soil",
    "Water",
    "Snow",
    "Herbaceous Vegetation",
    "Agricultural Land",
    "Plowed Land",
    "Vineyard",
    "Deciduous",
    "Coniferous",
    "Brushwood",
    "Clear Cut",
    "Ligneous",
    "Mixed",
    "Undefined"
]

ACTIVE_CLASSES = list(range(15))
IGNORED_CLASSES = [15, 16, 17, 18]

CLASS_COLORS = [
    (219, 94, 86),    # Building - red
    (219, 159, 94),   # Greenhouse - orange
    (0, 191, 255),    # Swimming Pool - cyan
    (192, 192, 192),  # Impervious Surface - gray
    (165, 124, 82),   # Pervious Surface - brown
    (210, 180, 140),  # Bare Soil - tan
    (0, 0, 255),      # Water - blue
    (255, 255, 255),  # Snow - white
    (144, 238, 144),  # Herbaceous Vegetation - light green
    (255, 255, 0),    # Agricultural Land - yellow
    (139, 69, 19),    # Plowed Land - dark brown
    (128, 0, 128),    # Vineyard - purple
    (34, 139, 34),    # Deciduous - forest green
    (0, 100, 0),      # Coniferous - dark green
    (154, 205, 50),   # Brushwood - yellow green
    (64, 64, 64),     # Clear Cut - dark gray
    (96, 96, 96),     # Ligneous - medium gray
    (128, 128, 128),  # Mixed - gray
    (0, 0, 0)         # Undefined - black
]

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_MIN = 1e-6

BASE_FILTERS = 32
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
