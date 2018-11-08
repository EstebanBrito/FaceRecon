import cv2
import numpy as np
import os


def drawRectangleText(img, rect, text):
    """Draw a rectangle with the given coordinates (rect) in the image"""
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    return img


def validateImgs(folder_path="training-data/test"):
    """Validate if the images inside folder are valid for
    facial recognition (imgs have at least one face,
    min/max size, etc). """

    imgs = []
    valid_imgs_paths = []
    non_valid_imgs = []

    # Reading paths for images inside chosen folder
    imgs_names = os.listdir(folder_path)

    # READING IMAGES
    for name in imgs_names:
        img_path = folder_path + "/" + name
        image = cv2.imread(img_path)
        imgs.append(image)

    # Creating face detector (using lbp cascade to simulate detection with video streaming)
    face_detector = cv2.CascadeClassifier("xml-files/lbpcascades/lbpcascade_frontalface.xml")

    for img in imgs:
        # STANDARDIZING IMAGES
        # Resizing images (shortest size should be 500px)
        if img.shape[0] < img.shape[0]:  # Height is shorter
            factor = 500/img.shape[0]
        else:  # Width is shorter
            factor = 500/img.shape[1]
        img = cv2.resize(src=img, dsize=(0, 0), fx=factor, fy=factor)

        # Tweaking images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # DETECTING AND SHOWING IMAGES
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

        # Validates if there are faces inside img
        if len(faces) != 1:
            non_valid_imgs.append(img)
        else:
            # Manipulating and showing valid images
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            valid_imgs_paths.append()

            cv2.imshow("Valid image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Showing non-valid images
    for img in non_valid_imgs:
        cv2.imshow("NON-VALID IMAGE", img)
        cv2.waitKey(0)
        # Perform action (change imgs, touch selected face (if multiple faces exists))
        cv2.destroyAllWindows()


def trainModel():
    print("Preparing data...")

    # TEMP Delete previous model file
    if os.path.isfile("model.yml"):
        os.remove("model.yml")
        os.remove("training-data/profiles.txt")

    # Lists that relates a face with its label, and a label/number with a name
    faces, labels, numbers, names = prepareTrainingData("training-data")

    print("Data prepared")
    print()

    # Results of training preparation
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    for number in numbers:
        print(number, end="   -   ")
    print()
    for name in names:
        print(name, end="   -   ")
    print()

    # Creating our face recognizer and training it
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    # Saving trained model
    face_recognizer.save("model.yml")
    # Saving face recognition profiles
    file = open("training-data/profiles.txt", "w")
    for i in range(len(numbers)):
        file.write(str(numbers[i]) + "-" + names[i] + "\n")
    file.close()


def detectFace(img):
    """Returns a cropped gray image containing the face detected in the
    given image. The coords of the face are also returned"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Creating face detector (using haar cascade for better accuracy)
    classifier = cv2.CascadeClassifier("xml-files/haarcascades/haarcascade_frontalface_default.xml")

    face = classifier.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # If no faces are detected don't return anything
    if len(face) == 0:
        return None, None

    # It is known that there will be only one face, extract the face area
    (x, y, w, h) = face[0]

    return gray[y:y + w, x:x + h], face[0]


def predict(test_img, recognizer, subjects):
    """Recognizes the person in the image and marks it with
    a rectangle and his/her name"""
    # Making a security copy of the img
    img = test_img

    # Detect face from the image
    face, rect = detectFace(img)

    # Predict the image using our face recognizer
    label = recognizer.predict(face)

    print(label)  # DEBUG

    # Get name of label (first elem of tuple, an integer) returned by face recognizer
    if label[1] < 60:  # If confidence is small enough
        if label[0] in subjects:
            label_text = subjects[label[0]]
        else:
            label_text = "Not registered"
    else:
        label_text = "Unknown"

    # Mark down the image with a rectangle and text
    img = drawRectangleText(img, rect, label_text)

    return img


def prepareTrainingData(data_folder_path="training-data"):
    """Reads training images path and returns two lists that relate
    an img with an integer (label for ech person to recognize)"""

    # Get dirs in data folder
    dirs = os.listdir(data_folder_path)

    # Lists for relations img-number
    faces = []
    labels = []
    # Lists for relation number-person
    numbers = []
    names = []

    # Go through each directory and read images within it
    for dir_name in dirs:
        # Ignoring folder that don't start with 's'
        if not dir_name.startswith("s"):
            continue

        # Building dir path for later reading of imgs within it
        # Sample: subject_dir_path = "training-data/s1"
        subject_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_path)

        # Getting label of current directory
        label = int(dir_name.replace("s", ""))
        numbers.append(label)

        # Getting name of the person whose faces are inside directory
        if not os.path.isfile(subject_path + "/name.txt"):
            print("Name for person in " + subject_path + " is required")
            exit(0)
        # Reading name from file inside dir
        file = open(subject_path + "/name.txt")
        name = file.read()
        name = name.replace("\n", "")
        file.close()
        names.append(name)

        # Go through each image path, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:
            # Ignore system files like .DS_Store and name.txt
            if image_name.startswith(".") or image_name.endswith(".txt"):
                continue

            # Build image path
            # Should be smth like: image path = "training-data/s1/1.jpg"
            image_path = subject_path + "/" + image_name

            # Read image
            image = cv2.imread(image_path)

            # Resizing images (shortest size should be 500px)
            if image.shape[0] < image.shape[0]:  # Height is shorter
                factor = 500 / image.shape[0]
            else:  # Width is shorter
                factor = 500 / image.shape[1]
            image = cv2.resize(src=image, dsize=(0, 0), fx=factor, fy=factor)

            # Detect face and add original and variants to training data
            face, rect = detectFace(image)
            if face is not None:
                # Add original pair
                faces.append(face)
                labels.append(label)

                # Get shortest shape
                if face.shape[0] < face.shape[0]:  # Height is shorter
                    shortest = face.shape[0]
                else:  # Width is shorter
                    shortest = face.shape[1]

                # Resize and add additional pairs
                for i in range(4):
                    # Get factor to resize to 60, 90, 120, 150px
                    factor = (60 + 30*i) / shortest
                    new_face = cv2.resize(src=face, dsize=None, fx=factor, fy=factor)
                    faces.append(new_face)
                    labels.append(label)

    return faces, labels, numbers, names


def showCurrentProfiles():
    if os.path.isfile("training-data/profiles.txt"):
        # Read profiles from file
        print("Leyendo perfiles")
        file = open("training-data/profiles.txt")
        for line in file:
            print(line, end="")
    else:
        print("No existe perfil alguno")


def startRecon():
    # DEFINING PARAMETERS (for better performance)
    min_face_size = 50  # 40 is good for PiCamera detection up to 4 meters
    max_face_size = 150

    # LOADING RESOURCES
    # Relations number-person
    subjects = {}
    if not os.path.isfile("training-data/profiles.txt"):
        print("No se encontro archivo de perfiles")
        exit(0)
    file = open("training-data/profiles.txt", "r")
    for line in file:
        line = line.replace("\n", "")
        subjects[int(line[0])] = line.replace(line[0] + "-", "")
    file.close()

    # Trained model
    if not os.path.isfile("training-data/profiles.txt"):
        print("No se encontro archivo de modelo")
        exit(0)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("model.yml")

    # Cascade classifier (using lbp for faster performance)
    cascade = cv2.CascadeClassifier('xml-files/lbpcascades/lbpcascade_frontalface.xml')

    # Video stream
    video = cv2.VideoCapture(0)

    # READING VIDEO STREAM
    while True:
        return_value, frame = video.read()
        if return_value == 0:  # If frame is empty, pick next frame
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecting faces in frame
        faces = cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(min_face_size, min_face_size),
            maxSize=(max_face_size, max_face_size)
        )

        # PROCESSING EACH IMAGE
        for (x, y, h, w) in faces:
            # Crop face
            cropped_face = gray_frame[y:y + w, x:x + h]

            # Predict face
            label = model.predict(cropped_face)

            # Get name of label (first elem of tuple, an integer) returned by face recognizer
            if label[1] < 100:  # If confidence is small enough ()
                if label[0] in subjects:
                    name = subjects[label[0]]
                else:
                    name = "Not registered"
            else:
                name = "Unknown"

            confidence = 100 - label[1]
            label_text = name + " - " + format(confidence, ".2f") + "%"

            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        # Draw rectangles for developer mode
        cv2.rectangle(frame, (0, 0), (0 + max_face_size, 0 + max_face_size), (255, 0, 0))  # Max size
        cv2.rectangle(frame, (max_face_size, 0), (max_face_size + min_face_size, 0 + min_face_size), (0, 0, 255))  # Min size

        # Display the resulting frame
        cv2.imshow('Video feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()


def test():
    # Relations number-person
    subjects = {}
    file = open("training-data/profiles.txt", "r")
    for line in file:
        line = line.replace("\n", "")
        subjects[int(line[0])] = line.replace(line[0] + "-", "")
    file.close()

    # Trained model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("model.yml")

    # Getting image paths from test-data
    img_names = os.listdir("test-data")

    for img_name in img_names:
        if not os.path.isfile("test-data/" + img_name):  # If not an image, skip it
            continue

        img = cv2.imread("test-data/" + img_name)

        # Resizing images (shortest size should be 500px)
        if img.shape[0] < img.shape[0]:  # Height is shorter
            factor = 500 / img.shape[0]
        else:  # Width is shorter
            factor = 500 / img.shape[1]
        img = cv2.resize(src=img, dsize=(0, 0), fx=factor, fy=factor)

        predimg = predict(img, model, subjects)

        cv2.imshow("Image", predimg)
        cv2.waitKey()
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
            # Temporary options(developer mode)
            print("[ 8 ] --- Testear modelo con imgs (de test-data)")
            print("[ 9 ] --- Validar nuevas imgs (de training-data/test)")
            print()
            op = int(input("Ingresa el numero de tu eleccion: "))
            print()

        if op == 1:
            if os.path.isfile("model.yml"):
                # Start facial recognition
                print("Reconociendo")
                print()
                startRecon()
            else:
                # There's no model. Train and then recon
                print("Entrenando y reconociendo")
                print()
                trainModel()
                startRecon()
        elif op == 2:
            # Stop facial recognition
            print("Deteniendo recon facial")
            print()
        elif op == 3:
            print("Entrenando modelo")
            print()
            trainModel()
            print()
        elif op == 4:
            print("Accediendo a perfiles")
            print()
        elif op == 5:
            print("Agregando perfil")
            print()
        elif op == 6:
            print("Removiendo perfil")
            print()
        elif op == 7:
            exit(0)
        elif op == 8:
            print("Testeando")
            test()
            print()
        elif op == 9:
            print("Validando")
            validateImgs()
            print()
        else:
            print("Opcion no valida")