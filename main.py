import cv2
import numpy as np
import os
import shutil
import time


def resizeDyn(refrncSide=None, size=None, factor=None):
    """Returns a resized version depending on different conditions"""
    pass


def convertToGray(img):
    """Returns a gray scale version of given img. Used when detecting faces"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def drawRectangleText(img, x, y, w, h, text):
    """Draw a rectangle with the given coordinates (rect) in the image"""
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    return img


def showCurrentProfiles():
    if os.path.isfile("model/profiles.txt"):
        # Read profiles from file
        print("Leyendo perfiles")
        file = open("model/profiles.txt")
        for line in file:
            print(line, end="")
        file.close()
    else:
        print("No existe perfil alguno")


def getNumbers(file):
    """Returns a list of current names founded in profiles file"""
    numbers = []
    if os.path.isfile(file):
        # Read profile names from file
        profiles = open(file, "r")
        for line in profiles:
            # Get only names from profiles file
            number = int(line[:1])
            numbers.append(number)
        profiles.close()
        return numbers
    else:
        return numbers


def getNames(file):
    """Returns a list of current names founded in profiles file"""
    names = []
    if os.path.isfile(file):
        # Read profile names from file
        profiles = open(file, "r")
        for line in profiles:
            # Get only names from profiles file
            name = line[2:len(line) - 1]
            names.append(name)
        profiles.close()
        return names
    else:
        return names


def cropAndSaveFaces(folder_path="training-data/temp"):
    """Save cropped faces from images inside folder_path.
    Used whe adding a new profile to face recognition model"""

    # List for images that require post processing
    non_valid_imgs_paths = []

    # Building path for folder where cropped faces imgs will be saved
    cropped_faces_folder = folder_path + "/valid-imgs"

    # Validates that a brand new folder is available to storage cropped faces
    if os.path.isdir(cropped_faces_folder):
        shutil.rmtree(cropped_faces_folder)
        os.mkdir(cropped_faces_folder)
    else:
        os.mkdir(cropped_faces_folder)

    # Reading names of the images inside chosen folder
    imgs_names = os.listdir(folder_path)

    # Creating face detector (using haar cascade for better detection accuracy)
    face_detector = cv2.CascadeClassifier("xml-files/haarcascades/haarcascade_frontalface_default.xml")

    # READING IMAGES
    # Counter for current number of saved cropped faces
    crops = 0

    for img_name in imgs_names:
        # Build path for next image to read
        img_path = folder_path + "/" + img_name

        # Skip file if its a folder (not a file)
        if not os.path.isfile(img_path):
            continue

        # Read image
        img = cv2.imread(img_path)

        # TWEAKING IMAGE
        # Resise image (shortest size should be 500px)
        if img.shape[0] < img.shape[0]:  # Height is shorter
            factor = 500/img.shape[0]
        else:  # Width is shorter
            factor = 500/img.shape[1]
        img = cv2.resize(src=img, dsize=(0, 0), fx=factor, fy=factor)

        # Convert image to gray scale (for better detection accuracy)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # DETECTING, CROPPING AND SAVING IMAGES
        # Detecting faces
        faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

        # If there is only one face inside img, proceed to crop
        if len(faces) == 1:
            (x, y, w, h) = faces[0]

            # Draw over and show img (uncomment if you need feedback)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow("Valid image number " + str(crops), img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Crop face from gray_img
            cropped_face = gray_img[y:y + w, x:x + h]
            # Build path to save img
            crops += 1
            cropped_img_name = cropped_faces_folder + "/" + str(crops) + ".jpg"
            # Save img
            cv2.imwrite(cropped_img_name, cropped_face)
        else:
            # Add path for posterior handling (manual validation, weaker detection, etc)
            non_valid_imgs_paths.append(img_path)

    # FINISH
    # Showing non-valid images (uncomment if you need feedback)
    # for img_path in non_valid_imgs_paths:
    #    print(img_path)
    #    img = cv2.imread(img_path)
    #    cv2.imshow("NON-VALID IMAGE", img)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

    print("Cropped " + str(crops) + " faces")


def getMediaFromImages():
    print("Pon las imgs manualmente en training-data/test. Espera 20s")
    time.sleep(20)


def getMediaFromWebcam(cropped_faces_folder="training-data/temp"):
    # Performance parameters
    min_face_size = 50
    max_face_size = 200

    # Loading face detector
    face_detector = cv2.CascadeClassifier("xml-files/haarcascades/haarcascade_frontalface_default.xml")

    # Loading video feed
    video = cv2.VideoCapture(0)

    # Counter for current frame
    crops = 0
    current_frame = 0
    cropping_is_active = False

    while True:
        # Read video frame by frame
        value, frame = video.read()
        # If frame is empty, pick next frame
        if value == 0:
            continue

        # DETECTING FACES
        # Convert frame to gray scale for better detection accuracy
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecting faces in frame
        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(min_face_size, min_face_size),
            maxSize=(max_face_size, max_face_size)
        )

        # PROCESSING EACH FACE IN FRAME
        for (x, y, h, w) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display resulting frame
        cv2.imshow('Video feed', frame)

        # CROPPING FACES
        # If cropping mode is active, crop face from frame and save it
        if cropping_is_active:
            current_frame += 1
            # If there is only one face, crop and save it
            if len(faces) == 1 and current_frame % 10 == 0:
                (x, y, h, w) = faces[0]
                cropped_face = gray_frame[y:y + w, x:x + h]

                # Build path to save img
                crops += 1
                cropped_img_path = cropped_faces_folder + "/" + str(current_frame) + ".jpg"
                # Save img
                cv2.imwrite(cropped_img_path, cropped_face)

        # Recognition will stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if cropping_is_active:
                print("")
                break
            else:
                cropping_is_active = True
                counter = 0
                crops = 0
            break

    # When everything is done, release resources (webcam or RPi stream)
    video.release()
    cv2.destroyAllWindows()


def addProfile(profiles_file="model/profiles.txt"):
    # VALIDATING
    # Get current profiles names
    names = getNames(profiles_file)

    # Name of new profile
    profile_name = ""

    # Validate name
    is_valid = False
    while not is_valid:
        profile_name = input("Ingrese el nombre de la persona: ")

        if profile_name in names:
            print("Nombre invalido")
        else:
            print("Nombre valido")
            is_valid = True

    # PROCESSING MEDIA
    # Validates that there is a folder to store incoming media
    if not os.path.isdir("training-data/temp"):
        os.mkdir("training-data/temp")

    # [Code to get media (from gallery, video sample, webcam) goes here]
    getMediaFromImages()

    # Detect and crop faces from given media
    cropAndSaveFaces()

    # Refresh status of current recognition model ("Outdated" means it needs retraining)
    file = open("model/status.txt", "w")
    file.write("Outdated")
    file.close()

    # SAVING PROFILE
    # Get number for new profile (smallest integer available)
    numbers = getNumbers(profiles_file)
    smallest = 1
    for i in sorted(numbers):
        if smallest == i:
            smallest += 1
        else:
            break
    profile_number = smallest

    # Build path for profile
    profile_path = "training-data/s" + str(profile_number)
    # Create folder for profile
    os.mkdir(profile_path)

    # Move collected faces to profile folder
    files = os.listdir("training-data/temp/valid-imgs")
    for file in files:
        shutil.move("training-data/temp/valid-imgs/" + file, profile_path)

    # Make new name file for profile
    file = open(profile_path + "/name.txt", "w")
    file.write(profile_name)
    file.close()

    # FINISH
    # Clean validation imgs folder
    shutil.rmtree("training-data/temp")


def prepareTrainingData(data_folder_path="training-data"):
    """Reads training images and returns two lists that relate a face
    with a label and two lists that relates a label/number with a person"""

    # Lists for relations img-number
    faces = []
    labels = []
    # Lists for relation number-person
    numbers = []
    names = []

    # Get folders names in data folder
    folder_names = os.listdir(data_folder_path)
    # Go through each directory and read images inside them
    for folder_name in folder_names:
        # Ignore anything that's not a folder and don't start with 's'
        if not os.path.isdir(folder_name) and not folder_name.startswith("s"):
            continue

        # Building dir path for later reading of imgs within it
        # Sample: subject_dir_path = "training-data/s1"
        subject_path = data_folder_path + "/" + folder_name

        # Getting label of current folder
        label = int(folder_name.replace("s", ""))
        numbers.append(label)

        # Validating that current folder has an owner (stated in name.txt)
        if not os.path.isfile(subject_path + "/name.txt"):
            print("Name for person in " + subject_path + " is required")
            exit(0)
        # Reading name from file inside current folder
        file = open(subject_path + "/name.txt")
        name = file.read()
        name = name.replace("\n", "")
        names.append(name)
        file.close()

        # Get names of imgs inside current folder
        subject_images_names = os.listdir(subject_path)
        # Add every cropped face image to list of faces
        for image_name in subject_images_names:
            # Ignore files that aren't images
            if not image_name.endswith(".jpg") and not image_name.endswith(".jpeg") and not image_name.endswith(".png"):
                continue

            # Build image path (smth like: image path = "training-data/s1/1.jpg")
            image_path = subject_path + "/" + image_name

            # Read image
            face = cv2.imread(image_path)
            face = convertToGray(face)

            # Add original pair
            faces.append(face)
            labels.append(label)

            # Adding more images for training
            # Get shortest shape
            if face.shape[0] < face.shape[0]:  # Height is shorter
                shortest = face.shape[0]
            else:  # Width is shorter
                shortest = face.shape[1]
            # Resize and add additional pairs
            for i in range(4):
                # Get factor to resize shortest size to 60, 90, 120, 150px
                factor = (60 + 30*i) / shortest
                new_face = cv2.resize(src=face, dsize=None, fx=factor, fy=factor)
                # Add additional pairs
                faces.append(new_face)
                labels.append(label)

    return faces, labels, numbers, names


def trainModel():
    """Generate face recognition model files using current training data"""
    print("Preparing data...")

    # Delete previous model data
    if os.path.isfile("model.yml"):
        os.remove("model.yml")
    if os.path.isfile("training-data/profiles.txt"):
        os.remove("training-data/profiles.txt")

    # Lists that relates a face with its label, and a label/number with a name
    faces, labels, numbers, names = prepareTrainingData("training-data")

    print("Data prepared")
    print()

    # Results of training preparation
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    print("Relations:")
    for i in range(len(numbers)):
        print(str(numbers[i]) + " - " + names[i])
    print()

    # Creating our face recognizer and training it
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    # Saving trained model
    face_recognizer.save("model/model.yml")
    # Saving face recognition profiles
    file = open("model/profiles.txt", "w")
    for i in range(len(numbers)):
        file.write(str(numbers[i]) + "-" + names[i] + "\n")
    file.close()
    # Updating status
    file = open("model/status.txt", "w")
    file.write("Updated")
    file.close()


def performPrediction(face, recognizer, subjects):
    """Recognizes the face of a person in the image and
    returns information about that person"""

    # Recognize face
    # Note: predict() returns label=(int number, double confidence)
    prediction = recognizer.predict(face)

    # Search person who it's related to the number returned by predict()...
    if prediction[1] < 100:  # ...if confidence is small enough
        if prediction[0] in subjects:  # ... and if that number was registered in profiles.txt
            name = subjects[prediction[0]]
        else:
            name = "Not registered"
    else:
        name = "Unknown"  # otherwise, its an unknown person

    # Build text to be draw in the image (with confidence
    # value converted to percentage)
    confidence = 100 - prediction[1]
    recognition_info = name + " - " + format(confidence, ".2f") + "%"

    return recognition_info


def loadSubjects():
    relations = {}

    if not os.path.isfile("model/profiles.txt"):
        print("No se encontro archivo de perfiles")
        exit(0)
    file = open("model/profiles.txt", "r")
    for line in file:
        line = line.replace("\n", "")
        relations[int(line[0])] = line.replace(line[0] + "-", "")
    file.close()

    return relations


def loadModel():
    if not os.path.isfile("model/model.yml"):
        print("No se encontro archivo de modelo")
        exit(0)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("model/model.yml")

    return face_recognizer


def startRecon():
    # DEFINING PARAMETERS (for best performance)
    min_face_size = 70  # (50-150) is good for PiCamera detection up to 4 meters
    max_face_size = 200

    # LOADING RESOURCES
    # Relations number-person (smth like {1: "Adolfo", 2: "Esteban", 3: "David"})
    subjects = loadSubjects()
    # Trained model
    model = loadModel()
    # Cascade classifier (using lbp for fast performance)
    cascade = cv2.CascadeClassifier('xml-files/lbpcascades/lbpcascade_frontalface.xml')
    # Video stream (here we can capture an RPi stream instance)
    video = cv2.VideoCapture(0)

    # READING VIDEO
    while True:
        # Read video frame by frame (here we can ask for every image the RPi streams/sends)
        return_value, frame = video.read()
        if return_value == 0:  # If frame is empty, pick next frame
            continue

        # Convert frame to gray scale for better detection accuracy
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecting faces in frame
        faces = cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(min_face_size, min_face_size),
            maxSize=(max_face_size, max_face_size)
        )

        # PROCESSING EACH FACE IN FRAME
        for (x, y, h, w) in faces:
            # Crop face
            cropped_face = gray_frame[y:y + w, x:x + h]
            # Perform recognition of cropped face
            recognition_info = performPrediction(cropped_face, model, subjects)
            # Draw rectangle and text
            frame = drawRectangleText(frame, x, y, h, w, recognition_info)

        # Draw rectangles indicating smallest and biggest space that can be detected as a face
        # (Uncomment if you need to calibrate the camera manually)
        cv2.rectangle(frame, (0, 0), (0 + min_face_size, 0 + min_face_size), (0, 0, 255))  # Min size
        cv2.rectangle(frame, (0, 0), (0 + max_face_size, 0 + max_face_size), (255, 0, 0))  # Max size

        # Display resulting frame
        cv2.imshow('Video feed', frame)

        # Recognition will stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release resources (webcam or RPi stream)
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        op = 0
        while op < 1 or op > 9:
            # Final version menu options
            print("MENU DE RECON FACIAL:")
            print("[ 1 ] --- Iniciar reconocimiento facial")
            print("[ 2 ] --- Detener reconocimiento facial")
            print("[ 3 ] --- Entrenar modelo")
            print("[ 4 ] --- Ver perfiles faciales del modelo")
            print("[ 5 ] --- Agregar perfiles faciales")
            print("[ 6 ] --- Remover perfiles faciales")
            print("[ 7 ] --- Salir")
            # Temporary options (developer mode)
            print("[ 8 ] --- Validar nuevas imgs (de training-data/test)")
            print()
            op = int(input("Ingresa el numero de tu eleccion: "))
            print()

        if op == 1:
            # Validates status of current face recognition model
            if os.path.isfile("model/status.txt"):
                file = open("model/status.txt", "r")
                status = file.read()
                status = status.replace("\n", "")
                if status == "Updated":
                    if os.path.isfile("model/model.yml"):
                        # Start facial recognition
                        print("Reconociendo")
                        print()
                        startRecon()
                        print()
                    else:
                        # There's no model. Train and then recon
                        print("Entrenando y reconociendo")
                        print()
                        trainModel()
                        startRecon()
                else:
                    # Model is out to date. Train and then recon
                    print("Actualizando y reconociendo")
                    print()
                    trainModel()
                    startRecon()
            else:
                print("No existe status")
                exit(0)
        elif op == 2:
            # Stop facial recognition
            print("Deteniendo recon facial")
            # [Facial recon stopping code goes here]
            print()
        elif op == 3:
            print("Entrenando modelo")
            print()
            trainModel()
            print()
        elif op == 4:
            print("Accediendo a perfiles")
            showCurrentProfiles()
            print()
        elif op == 5:
            print("Agregando perfil")
            addProfile()
            print()
        elif op == 6:
            print("Removiendo perfil")
            # [Facial profile remotion goes here]
            print()
        elif op == 7:
            exit(0)
        elif op == 8:
            print("Validando")
            cropAndSaveFaces()
            print()
        elif op == 9:
            print()
            getMediaFromWebcam()
            print()
        else:
            print("Opcion no valida")
            print()