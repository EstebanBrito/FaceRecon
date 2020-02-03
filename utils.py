import cv2
import os
from settings import MIN_SIZE, MAX_SIZE, MODEL_FOLDER, DATA_FOLDER, NAME_FILE

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

def drawBoundaries(frame, min_size=MIN_SIZE, max_size=MAX_SIZE):
  '''Draws smallest and biggest space that can be detected as a face'''
  cv2.rectangle(frame, (0, 0), (min_size, min_size), BLUE)
  cv2.rectangle(frame, (0, 0), (max_size, max_size), RED)

def drawRectangleText(img, x, y, w, h, text):
  '''Draw a rectangle and some text at the given coordinates in the image'''
  cv2.rectangle(img, (x, y), (x + w, y + h), GREEN)
  cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, GREEN)

def setupFolderStr():
  '''Creates folder structure not included in the project'''
  if not os.path.isdir(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
  if not os.path.isdir(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def loadLabels(data_path=DATA_FOLDER):
  labels = []
  folder_names = os.listdir(data_path)
  for folder_name in folder_names:
    label = int(folder_name.replace('s', ''))
    labels.append(label)
  return labels

def loadNames(data_path=DATA_FOLDER):
  names = []
  folder_names = os.listdir(data_path)
  for folder_name in folder_names:
    name_path = os.path.join(data_path, folder_name, NAME_FILE)
    file = open(name_path)
    name = file.read()
    name = name.strip()
    file.close()
    names.append(name)
  return names

def loadUntrainedProfiles(data_path=DATA_FOLDER):
  profiles = {}
  folder_names = os.listdir(data_path)
  for folder_name in folder_names:
    # Get label
    label = int(folder_name.replace('s', ''))
    # Get name
    name_path = os.path.join(data_path, folder_name, NAME_FILE)
    file = open(name_path)
    name = file.read()
    name = name.strip()
    file.close()
    # Append
    profiles[label] = name
  return profiles