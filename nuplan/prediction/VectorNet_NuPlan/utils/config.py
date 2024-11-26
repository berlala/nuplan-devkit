import os

# Nuplan related
DATA_PATH = '/home/bolin/Projects/shenlan/nuplan/dataset/nuplan-v1.1/splits/mini'
MAP_PATH  = '/home/bolin/Projects/shenlan/nuplan/dataset/maps'
SAVE_PATH = '/home/bolin/Projects/shenlan/nuplan-devkit/nuplan/prediction/VectorNet_NuPlan/processed_data'
MAP_VERSION = "nuplan-maps-v1.0"
SCENARIOS_PER_TYPE = 2000
TOTAL_SCENARIOS = 6000

# Vectornet Related
DISCRETE_SIZE = 10
PAST_TIME_HORIZON = 4 # [seconds]
NUM_PAST_POSES = DISCRETE_SIZE * PAST_TIME_HORIZON 
FUTURE_TIME_HORIZON = 3 # [seconds]
NUM_FUTURE_POSES = DISCRETE_SIZE * FUTURE_TIME_HORIZON
NUM_AGENTS = 10

LANE_NUM = 40
ROUTE_LANES_NUM = 10
CROSSWALKS_NUM = 5

LANE_POINTS_NUM = 51
ROUTE_LANES_POINTS_NUM = 51
CROSSWALKS_POINTS_NUM = 31

NODE_SIZE = CROSSWALKS_NUM * (CROSSWALKS_POINTS_NUM - 1) \
            + LANE_NUM * (LANE_POINTS_NUM - 1) \
            + ROUTE_LANES_NUM * (ROUTE_LANES_POINTS_NUM - 1) \
            + (NUM_AGENTS + 1) * NUM_PAST_POSES

NUM_GRAPH = LANE_NUM + ROUTE_LANES_NUM + CROSSWALKS_NUM + NUM_AGENTS + 1

QUERY_RADIUS = 60

# Training related
TRAIN_PATH = os.path.join(SAVE_PATH, 'train')
TEST_PATH = os.path.join(SAVE_PATH, 'test')
WEIGHT_PATH = '${your_local_path}/VectorNet_NuPlan/trained_params'
SEED = 999
EPOCHS = 50
BATCH_SIZE = 1

LEARNING_RATE = 0.01
IN_CHANNELS = 8
OUT_CHANNELS = NUM_FUTURE_POSES * 2

DECAY_LR_EVERY = 10
DECAY_LR_FACTOR = 0.3

SHOW_EVERY = 100
VAL_EVERY = 5

VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1