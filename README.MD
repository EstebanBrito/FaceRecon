# Face Recognition for SmartCar

## Tools used
* Python 3.6
* virtualenv
* OpenCV (opencv-contrib-python) 4.1.2.30
* Ubuntu 18.04

# Setup
## Clone project
```bash
$ git clone repositoryURL
```
## Virtual Environment
This section is highly recommended, but not mandatory. Creating a virtual env. helps the FaceRecon encapsulate everything it needs into the folder, avoiding screwing up the global Python package system of your computer and other bugs.   

In case you dont have virtualenv package installed, do:   
```bash
$ pip3 install virtualenv
```

Create a virtual env.   
```bash
$ python3 -m virtualenv env
```

Activate environment. Your prompt should be preceded by a (env) if all went OK.   
```bash
$ source env/bin/activate
```
## Dependencies installation
Just do:
``` bash
$ pip3 install -r requirements.txt
```
If installation fails, try installing manually:
``` bash
$ pip3 install opencv-contrib-python===4.1.2.30
```

# Execution
```bash
$ python3 main.py
```

# Others
## Known issues
1.-Although pip downloads the required packages, Python doesn't seem to recognize 'cv2' module from opencv-contrib-python.
