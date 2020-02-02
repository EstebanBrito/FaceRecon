import cv2
from time import sleep
from os import system, name, path

CLFR_FOLDER = path.join('cascades', 'haar')
CLFR_FILE = 'haarcascade_frontalface_default.xml'
CLFR_PATH = path.join(CLFR_FOLDER, CLFR_FILE)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def clear():
  if name == 'nt': # for windows
    _ = system('cls') 
  else: # for mac and linux(here, os.name is 'posix')
    _ = system('clear')

def waitAndClear(waitTime=1.5): 
  sleep(waitTime)
  clear()

def convertImgToGray(img):
    """Returns a gray scale version of given img. Used when detecting faces"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def drawRectangleText(img, x, y, w, h, text):
    """Draw a rectangle with the given coordinates (rect) in the image"""
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, GREEN, 2)
    return img