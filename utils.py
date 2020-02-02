import cv2
import os
from settings import MIN_SIZE, MAX_SIZE, MODEL_FOLDER, DATA_FOLDER

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