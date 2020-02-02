import cv2
from utils import CLFR_PATH, convertImgToGray, drawRectangleText, RED, BLUE, GREEN

MIN_FACE_SIZE = 100
MAX_FACE_SIZE = 250

def facialDetection(min_face_size=MIN_FACE_SIZE, max_face_size=MAX_FACE_SIZE):
  cascade = cv2.CascadeClassifier(CLFR_PATH)
  video = cv2.VideoCapture(0)

  while True:
    avaliable, frame = video.read()
    if avaliable == 0: # skip non avaliable frames
      continue

    grayFrame = convertImgToGray(frame)

    faces = cascade.detectMultiScale(
      grayFrame,
      scaleFactor=1.1, minNeighbors=8,
      minSize=(min_face_size, min_face_size),
      maxSize=(max_face_size, max_face_size)
    )
    
    # PROCESSING EACH FACE IN FRAME
    for (x, y, h, w) in faces:
      frame = drawRectangleText(frame, x, y, h, w, '')
    # Smallest and biggest space that can be detected as a face
    cv2.rectangle(frame, (0, 0), (0 + min_face_size, 0 + min_face_size), BLUE)  # Min size
    cv2.rectangle(frame, (0, 0), (0 + max_face_size, 0 + max_face_size), RED)  # Max siz
    
    cv2.imshow('Video feed', frame)
    # Recognition will stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  video.release()
  cv2.destroyAllWindows()