import cv2
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

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def clear():
  if os.name == 'nt': # for windows
    _ = os.system('cls') 
  else: # for mac and linux (here, os.name is 'posix')
    _ = os.system('clear')

def convertImgToGray(img):
  '''Returns a gray scale version of any given image. Used when detecting faces'''
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

def drawRectangleText(img, x, y, w, h, text):
  '''Draw a rectangle and some text at the given coordinates in the image'''
  cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
  cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, GREEN, 2)
  return img

def setupFolderStr():
  '''Creates folder structure not included in the project'''
  if not os.path.isdir(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
  if not os.path.isdir(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)