from time import sleep
from utils import clear, setupFolderStr
from detection import facialDetection, loadProfiles
from training import trainModel
from profile_handling import addProfile, removeProfile

if __name__ == "__main__":
  setupFolderStr()
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
    # TODO: Flush input
    op = int(input('Select an option: '))
    print()
    clear()
    try:
      if op==1:
        print('Starting facial detection')
        facialDetection()
      elif op==2:
        print('Starting facial recognition')
        facialDetection(mode=1)
      elif op==3:
        print('Starting model training')
        trainModel()
      elif op==4:
        print('MODEL PROFILES')
        profiles = loadProfiles()
        for key in profiles:
          print(f'{key} - {profiles[key]}')
      elif op==5:
        addProfile()
      elif op==6:
        removeProfile()
      elif op==7:
        exit(0)
      else:
        print('Unvalid option')
    except AssertionError as Argument:
      print('ATTENTION: Program cannot run due to this failed precondition:\n')
      print('*', Argument)
      print()
      print('Repair this error before using the program again')
      print()
      sleep(2)
    except Exception as Argument: # don't use BaseException
      print('ATTENTION: An unexpected exception occurred. Please, inform the developer')
      print(Argument) # DEBUG
      print()
      sleep(2)
    else:
      sleep(2)
      clear()

