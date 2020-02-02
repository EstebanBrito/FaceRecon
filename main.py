import cv2
from utils import *
from detection import facialDetection

if __name__ == "__main__":
  while True:
    print('FACE RECON MENU')
    print(' [1] --- Start facial detection')
    print(' [2] --- Start facial recognition')
    print(' [3] --- Train model')
    print(' [4] --- Check facial profiles')
    print(' [5] --- Add facial profile')
    print(' [6] --- Remove facial profile')
    print(' [7] --- Exit')
    print()
    op = int(input('Select an option: '))
    print()

    clear()
    if op==1:
      print('Starting facial detection')
      facialDetection()
    if op==7:
      print('Saliendo')
      waitAndClear()
      exit(0)
    if op not in range(1, 8):
      print('Unvalid option')
    waitAndClear()

