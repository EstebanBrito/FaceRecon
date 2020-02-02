import cv2
import numpy as np
import os
from utils import convertImgToGray
from settings import MODEL_PATH, PROFILES_PATH, DATA_FOLDER

NAME_FILE = 'name.txt'


def prepareTrainingData(data_folder_path=DATA_FOLDER):
  '''Reads training images and returns lists that relate a face
  with a label and a label with a person'''
  faces = []
  labels = []
  numbers = []
  names = []
  # Read every folder inside data_folder_path
  folder_names = os.listdir(data_folder_path)
  assert (len(folder_names)>0), f'There is no data in {data_folder_path}. Add some profiles first.'
  for folder_name in folder_names:
    # Ignore files and folders that don't start with 's'
    if not os.path.isdir(folder_name) and not folder_name.startswith("s"):
      continue
    # Build folder path (example: "training-data/s1")
    subject_path = os.path.join(data_folder_path, folder_name)
    # Get label (number) from folder name
    label = int(folder_name.replace('s', ''))
    numbers.append(label)
    # Get name from file (name.txt) inside the folder
    file = open(os.path.join(subject_path, NAME_FILE))
    name = file.read()
    name = name.replace('\n', '')
    names.append(name)
    file.close()
    # Read every img inside the folder and add its gray version to list
    images_names = os.listdir(subject_path)
    for image_name in images_names:
      # Ignore files that aren't images
      ext = image_name.split('.')[-1]
      if ext not in ('jpg', 'jpeg', 'png'):
        continue
      # Build image path (example: "training-data/s1/1.jpg")
      image_path = os.path.join(subject_path, image_name)
      # Read image
      face = cv2.imread(image_path)
      face = convertImgToGray(face)
      # Add original pair
      faces.append(face)
      labels.append(label)
      # Data augmentation: Add more data (resized images)
      # for i in range(4):
      #   # Get factor to resize to 60, 90, 120, 150px
      #   factor = (60 + 30*i) / face.shape[0]
      #   new_face = cv2.resize(src=face, dsize=None, fx=factor, fy=factor)
      #   # Add additional pairs
      #   faces.append(new_face)
      #   labels.append(label)
  return faces, labels, numbers, names


def trainModel():
  '''Generate face recognition model files using current training data'''
  print('Preparing data...')
  print()
  # Lists that relates a face with a label, and a label (number) with a name
  faces, labels, numbers, names = prepareTrainingData()
  print('Data prepared!')
  print(f'Total faces: {len(faces)}')
  print(f'Total labels: {len(labels)}')
  print('Relations:')
  for number, name in zip(numbers, names):
    print(f'{number} - {name}')
  print()
  
  # Train facial recognition model
  print('Training model...')
  model = cv2.face.LBPHFaceRecognizer_create()
  model.train(faces, np.array(labels))
  print('Model trained!')
  
  # Delete previous model and profiles
  print('Saving model...')
  if os.path.isfile(MODEL_PATH):
    os.remove(MODEL_PATH)
  if os.path.isfile(PROFILES_PATH):
    os.remove(PROFILES_PATH)
  # Save model and facial recognition profiles
  model.save(MODEL_PATH)
  file = open(PROFILES_PATH, 'w')
  for number, name in zip(numbers, names):
    file.write(f'{number}|{name}\n')
  file.close()
  print('Model saved!')
  print('Done!')
  # Updating status
  # file = open("model/status.txt", "w")
  # file.write("Updated")
  # file.close()