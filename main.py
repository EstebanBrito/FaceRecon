import cv2
import numpy as np
import os
import shutil


def convertToGray(img):
    """Returns a gray scale version of given img. Used when detecting faces"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def drawRectangleText(img, x, y, w, h, text):
    """Draw a rectangle with the given coordinates (rect) in the image"""
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    return img


def getNumbers(profiles_file_path="model/profiles.txt"):
    """Returns a list of current numbers founded in profiles file"""
    numbers = []
    if os.path.isfile(profiles_file_path):
        file = open(profiles_file_path, "r")
        for line in file:
            # Get only numbers from profiles file
            number = int(line[:1])
            numbers.append(number)
        file.close()
        return numbers
    else:
        return numbers


def getNames(profiles_file_path="model/profiles.txt"):
    """Returns a list of current names founded in profiles file"""
    names = []
    if os.path.isfile(profiles_file_path):
        file = open(profiles_file_path, "r")
        for line in file:
            # Get only names from profiles file
            name = line[2:len(line) - 1]
            names.append(name)
        file.close()
        return names
    else:
        return names


def getFacesFromWebcam(cropped_faces_path="training-data/temp/valid-imgs"):
    """Uses webcam to detect, crop and save faces"""
    # PERFORMANCE PARAMETERS
    min_face_size = 50
    max_face_size = 250
    frame_period = 15

    # FOLDER VALIDATION
    # Validates that a brand new folder is available to storage cropped faces
    if os.path.isdir(cropped_faces_path):
        shutil.rmtree(cropped_faces_path)
        os.mkdir(cropped_faces_path)
    else:
        os.mkdir(cropped_faces_path)

    # LOADING RESOURCES
    # Loading face detector
    face_detector = cv2.CascadeClassifier("xml-files/haarcascades/haarcascade_frontalface_default.xml")
    # Loading video feed
    video = cv2.VideoCapture(0)

    # PREPARATION
    # Instructions
    print("Press 'q' once to start recollecting images. The person"
          "should look straight to the camera and make different"
          "expressions. The person must not tilt his/her face to the "
          "sides.\n"
          "Press 'q' again to quit data recollection")

    # Counter for number of cropped faces
    crops = 0
    # Counter for number of captured frame
    current_frame = 0
    # Flag used to turn on face cropping mode
    cropping_is_active = False

    # READING FRAMES
    while True:
        # Read video frame by frame
        value, frame = video.read()
        # If frame is empty, pick next frame
        if value == 0:
            continue

        # Draw boundaries
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # DETECTING FACES AND DISPLAYING VIDEO
        # Convert frame to gray scale for better detection accuracy
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in frame
        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(min_face_size, min_face_size),
            maxSize=(max_face_size, max_face_size)
        )

        # Draw faces over frame
        for (x, y, h, w) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display resulting frame
        cv2.imshow('Video feed', frame)

        # CROPPING FACES
        # If cropping mode is active, count frames and...
        if cropping_is_active:
            current_frame += 1
            # ...if there is only one face, crop and save it
            if len(faces) == 1 and current_frame % frame_period == 0:
                (x, y, h, w) = faces[0]
                cropped_face = gray_frame[y:y + w, x:x + h]

                # Build path to save img
                crops += 1
                cropped_img_path = cropped_faces_path + "/" + str(crops) + ".jpg"
                # Save img
                cv2.imwrite(cropped_img_path, cropped_face)

        # Press 'q' to start face cropping. Press again to terminate recognition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if cropping_is_active:
                break
            else:
                cropping_is_active = True
                current_frame = 0
                crops = 0

    # When everything is done, release resources (webcam or RPi stream)
    video.release()
    cv2.destroyAllWindows()


def addProfile(model_data_path="model", media_folder_path="training-data/temp"):
    # NAME VALIDATION
    # Get current profiles names
    names = getNames(model_data_path + "/profiles.txt")

    # Validation loop
    profile_name = ""
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
    if not os.path.isdir(media_folder_path):
        os.mkdir(media_folder_path)
    # Validates that a brand new folder is available to storage cropped faces
    if os.path.isdir(media_folder_path + "/valid-imgs"):
        shutil.rmtree(media_folder_path + "/valid-imgs")
        os.mkdir(media_folder_path + "/valid-imgs")
    else:
        os.mkdir(media_folder_path + "/valid-imgs")

    # [Prompt to choose way get media (from gallery, video sample, webcam) goes here]
    getFacesFromWebcam()

    # SAVING PROFILE
    # Get number for new profile (smallest integer available)
    numbers = getNumbers(model_data_path + "/profiles.txt")
    smallest = 1
    for i in sorted(numbers):
        if smallest == i:
            smallest += 1
        else:
            break
    profile_number = smallest

    # Build path for profile
    new_profile_path = "training-data/s" + str(profile_number)
    # Create folder for profile
    os.mkdir(new_profile_path)

    # Move collected faces to profile folder
    files = os.listdir(media_folder_path + "/valid-imgs")
    for file in files:
        shutil.move(media_folder_path + "/valid-imgs/" + file, new_profile_path)

    # Make new name.txt file for profile
    file = open(new_profile_path + "/name.txt", "w")
    file.write(profile_name)
    file.close()

    # FINISH
    # Refresh status of current recognition model ("Outdated" means it needs retraining)
    file = open(model_data_path + "/status.txt", "w")
    file.write("Outdated")
    file.close()

    # Clean incoming media folder (for storage space saving)
    shutil.rmtree(media_folder_path)


def showCurrentProfiles(profiles_file_path="model/profiles.txt"):
    if os.path.isfile(profiles_file_path):
        # Read profiles from file
        print("PERFILES ACTUALES")
        file = open(profiles_file_path)
        for line in file:
            print(line, end="")
        file.close()
    else:
        print("No existe perfil alguno")


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
    print()

    # Lists that relates a face with a label, and a label (number) with a name
    faces, labels, numbers, names = prepareTrainingData("training-data")

    print("Data prepared!")
    print()

    # Results of training data preparation
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    print("Relations:")
    for i in range(len(numbers)):
        print(str(numbers[i]) + " - " + names[i])
    print()

    # Create face recognizer and train it
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    # Delete previous model data (will be replaced by new model file)
    if os.path.isfile("model/model.yml"):
        os.remove("model/model.yml")
    if os.path.isfile("model/profiles.txt"):
        os.remove("model/profiles.txt")

    # Save trained model
    face_recognizer.save("model/model.yml")

    # Save face recognition profiles
    file = open("model/profiles.txt", "w")
    for i in range(len(numbers)):
        file.write(str(numbers[i]) + "-" + names[i] + "\n")
    file.close()

    # Updating model status
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
        if prediction[0] in subjects:  # ... and if that number is registered in profiles.txt
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
    min_face_size = 50  # (50-150) is good for PiCamera detection up to 4 meters
    max_face_size = 200

    # LOADING RESOURCES
    # Relations number-person (smth like {1: "Fernando", 2: "Esteban", ...})
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
        # cv2.rectangle(frame, (0, 0), (0 + min_face_size, 0 + min_face_size), (0, 0, 255))  # Min size
        # cv2.rectangle(frame, (0, 0), (0 + max_face_size, 0 + max_face_size), (255, 0, 0))  # Max size

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
        while op < 1 or op > 6:
            # Final version menu options
            print("MENU DE RECON FACIAL:")
            print("[ 1 ] --- Iniciar reconocimiento facial")
            print("[ 2 ] --- Entrenar modelo")
            print("[ 3 ] --- Ver perfiles faciales del modelo")
            print("[ 4 ] --- Agregar perfiles faciales")
            print("[ 5 ] --- Salir")
            # Temporary options (developer mode)
            print("[ 6 ] --- Usar webcam para conseguir caras para entrenamiento")
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
                        print("Reconociendo...")
                        print()
                        startRecon()
                        print()
                    else:
                        # There's no model. Train and then recon
                        print("Entrenando y reconociendo...")
                        print()
                        trainModel()
                        startRecon()
                        print()
                else:
                    # Model is out to date. Train and then recon
                    print("Actualizando y reconociendo...")
                    print()
                    trainModel()
                    startRecon()
                    print()
            else:
                print("No existe status")
                exit(0)
        elif op == 2:
            print("Entrenando modelo...")
            print()
            trainModel()
            print()
        elif op == 3:
            print("Accediendo a perfiles...")
            showCurrentProfiles()
            print()
        elif op == 4:
            print("Agregando perfil...")
            addProfile()
            print()
        elif op == 5:
            exit(0)
        elif op == 6:
            print()
            getFacesFromWebcam()
            print()
        else:
            print("Opcion no valida")
            print()