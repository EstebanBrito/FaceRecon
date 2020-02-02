import cv2
import os
from utils import convertImgToGray, drawRectangleText, drawBoundaries, RED, BLUE, GREEN
from settings import CLFR_PATH, MODEL_PATH, PROFILES_PATH, MIN_SIZE, MAX_SIZE

DETECTION = 0
RECOGNITION = 1

def loadModel(path=MODEL_PATH):
  assert os.path.isfile(path), 'There is no model. Train a model first.'
  model = cv2.face.LBPHFaceRecognizer_create()
  model.read(path)
  return model

def loadProfiles(path=PROFILES_PATH):
  '''Returns a dictionary linking people's recognition ID with their names'''
  assert os.path.isfile(path), f'{path} does not exist. Train a model first.'
  profiles = {}
  file = open(path, 'r')
  for line in file:
    data = line.split('|')
    profiles[data[0]] = data[1].strip() # strip removes \n
  file.close()
  return profiles

def performPrediction(face, recognizer, profiles):
  '''Recognizes the face of a person in the image and
  returns information about that person'''
  # Note: predict() returns (int number, double confidence)
  label, confidence = recognizer.predict(face)
  label = str(label) # keys are string inside dicts
  if confidence < 100:
    if label in profiles.keys():
      name = profiles[label]
    else:
      name = 'Not registered'
  else:
    name = 'Unknown'
  return f'{name} - {format(confidence, ".2f")}'

def facialDetection(mode=DETECTION, min_size=MIN_SIZE, max_size=MAX_SIZE):
  if mode==RECOGNITION:
    model = loadModel()
    profiles = loadProfiles()
  cascade = cv2.CascadeClassifier(CLFR_PATH)
  video = cv2.VideoCapture(0)

  try:
    while True:
      avaliable, frame = video.read()
      if avaliable == 0: # skip non avaliable frames
        continue

      gray_frame = convertImgToGray(frame)

      faces = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1, minNeighbors=8,
        minSize=(min_size, min_size),
        maxSize=(max_size, max_size)
      )

      # PROCESSING EACH FACE IN FRAME
      for (x, y, h, w) in faces:
        if mode==DETECTION:
          drawRectangleText(frame, x, y, h, w, '')
        if mode==RECOGNITION:
          cropped_face = gray_frame[y:y + w, x:x + h]
          recon_info = performPrediction(cropped_face, model, profiles)
          drawRectangleText(frame, x, y, h, w, recon_info)

      drawBoundaries(frame)
      cv2.imshow('Video feed', frame)
      # Video stream will stop if 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  except Exception as Argument:
    raise Exception(Argument)
  finally:
    video.release()
    cv2.destroyAllWindows()

