import cv2
import os
import shutil
from utils import convertImgToGray, drawRectangleText, drawBoundaries, loadUntrainedProfiles
from settings import CLFR_PATH, DATA_FOLDER, NAME_FILE, MIN_SIZE, MAX_SIZE


def getFaces(profile_path, min_size=MIN_SIZE, max_size=MAX_SIZE):
  cascade = cv2.CascadeClassifier(CLFR_PATH)
  video = cv2.VideoCapture(0)
  crops = 0
  cropping_is_active = False
  current_frame = 0
  print('''Press 'q' once to start recollecting images. The person
    should look straight to the camera and make different expressions.
    The person must not tilt his/her face to the sides.
    Press 'q' again to quit data recollection''')

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

      for (x, y, h, w) in faces:
        drawRectangleText(frame, x, y, w, h, '')
      
      if cropping_is_active:
        current_frame += 1
        if len(faces)==1 and current_frame % 40 == 0:
          (x, y, h, w) = faces[0]
          cropped_face = gray_frame[y:y + w, x:x + h]
          crops += 1
          cropped_img_path = os.path.join(profile_path, f'{crops}.jpg')
          cv2.imwrite(cropped_img_path, cropped_face)

      drawBoundaries(frame)
      cv2.imshow('Video feed', frame)
      # Video stream will swap mode if 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        if cropping_is_active:
          break
        else:
          cropping_is_active = True
  except Exception as Argument:
    raise Exception(Argument)
  finally:
    video.release()
    cv2.destroyAllWindows()


def addProfile(data_folder=DATA_FOLDER):
  profiles = loadUntrainedProfiles()

  name = ''
  while True:
    name = input('Enter the name of the person: ')
    if name in profiles.items():
      print('That name is already in use')
    else:
      break

  labels = [int(x) for x in profiles.keys()]
  smallest = 1
  for i in sorted(labels):
    if smallest == i:
      smallest += 1
    else:
      break
  label = smallest
  
  profile_path = os.path.join(data_folder, f's{label}')
  os.makedirs(profile_path)

  file = open(os.path.join(profile_path, NAME_FILE), 'w')
  file.write(name)
  file.close()

  try:
    getFaces(profile_path)
  except Exception as Argument:
    shutil.rmtree(profile_path)
    raise Exception(Argument)


def removeProfile(data_folder=DATA_FOLDER):
  profiles = loadUntrainedProfiles()
  while True:
    print('EXISTING PROFILES (not necessarily in the model)')
    for key in profiles:
      print(f'{key} - {profiles[key]}')
    print()
    try:
      op = int(input('Select the number of the profile you want to delete. Press 0 to cancel: '))
    except:
      print('Select a number, please\n')
      continue
    if op==0:
      return
    if op in profiles.keys():
      break
    else:
      print('Select an existing profile')
      print()
  print('Deleting profile data...')
  profile_folder = os.path.join(data_folder, f's{op}')
  shutil.rmtree(profile_folder)
  print('Done!')
