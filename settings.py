import os

CLFR_FOLDER = os.path.join('cascades', 'haar')
CLFR_FILE = 'haarcascade_frontalface_default.xml'
CLFR_PATH = os.path.join(CLFR_FOLDER, CLFR_FILE)
MODEL_FOLDER = 'model'
PROFILES_FILE = 'profiles.txt'
MODEL_FILE = 'model.yml'
PROFILES_PATH = os.path.join(MODEL_FOLDER, PROFILES_FILE)
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILE)
DATA_FOLDER = 'training-data'

MIN_SIZE = 100
MAX_SIZE = 250
