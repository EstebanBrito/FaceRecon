import cv2
import numpy as np
import os


def validateImgs(folder_path="training-data/test"):
    """Validate if the images inside folder are valid for
    facial recognition (imgs have at least one face,
    min/max size, etc). """

    imgs_names = os.listdir(folder_path)
    imgs = []
    non_valid_imgs = []

    # Read imgs
    for name in imgs_names:
        img_path = folder_path + "/" + name
        print(img_path)  # DEBUG
        image = cv2.imread(img_path)
        imgs.append(image)

    faced = cv2.CascadeClassifier("xml-files/haarcascades/haarcascade_frontalface_default.xml")

    # Tweak imgs and detect faces
    for img in imgs:
        # Resizing images (shortest size should be 500px)
        print("Size: " + str(img.shape[1]) + "x" + str(img.shape[0]), end=" - ")

        factor = 0.75
        if img.shape[0] < img.shape[0]:  # Height is shorter
            factor = 500/img.shape[0]
        else:  # Width is shorter
            factor = 500/img.shape[1]
        img = cv2.resize(src=img, dsize=(0, 0), fx=factor, fy=factor)

        print("Size: " + str(img.shape[1]) + "x" + str(img.shape[0]))

        # Tweaking images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detecting faces
        faces = faced.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

        # Validates if there are faces inside img
        if len(faces) != 1:
            non_valid_imgs.append(img)
        else:
            # Manipulating and showing valid images
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
    # Two lists that relates a face with its label
    faces, labels, numbers, names = prepareTrainingData("training-data")

    print("Data prepared")
    print()

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

    # saving trained model
    face_recognizer.save("model.yml")
    # saving relations number-name


def detectFace(img):
    """Returns a cropped gray image containing the face detected in the
    given image. The coords of the face are also returned"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    classifier = cv2.CascadeClassifier('xml-files/lbpcascades/lbpcascade_frontalface.xml')

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
            # Sample: image path = training-data/s1/1.jpg
            image_path = subject_path + "/" + image_name

            # Read and display image
            image = cv2.imread(image_path)

            cv2.imshow("Training on image...", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Detect face, add faces and label to the lists
            face, rect = detectFace(image)
            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels, numbers, names


def showCurrentProfiles():
    if os.path.isfile("training-data/profiles.txt"):
        # Read profiles from file
        print("Leyendo perfiles")
    else:
        print("No existe perfil alguno")


def showProfileMenu():
    """Show a menu for profile management"""
    while True:
        op2 = 0
        while op2 < 1 or op2 > 4:
            print("ADMINISTRAR PERFILES")
            print("[ 1 ] --- Mostrar perfiles actuales")
            print("[ 2 ] --- Añadir perfil")
            print("[ 3 ] --- Eliminar perfil")
            print("[ 4 ] --- Regresar a menu principal")
            op2 = int(input("Ingresa el numero de tu eleccion: "))
            print()

        if op2 == 1:
            showCurrentProfiles()
            print()
        elif op2 == 2:
            print("Adding profile")
            print()
            validateImgs()
            print()
        elif op2 == 3:
            # Delete profile
            print("Deleting profile")
            print()
        elif op2 == 4:
            break


if __name__ == "__main__":
    while True:
        op = 0
        while op < 1 or op > 3:
            print("MENU DE RECON FACIAL:")
            print("[ 1 ] --- Iniciar Recon Facial")
            print("[ 2 ] --- Administrar perfiles faciales")
            print("[ 3 ] --- Salir")
            op = int(input("Ingresa el numero de tu eleccion: "))
            print()

        if op == 1:
            if os.path.isfile("model.yml"):
                # Start facial recognition
                print("Iniciando recon")
                print()
                # startRecon()
            else:
                # Train the model and then start recon
                print("Training and recon")
                print()
                trainModel()
                # startRecon()
        elif op == 2:
            showProfileMenu()
        elif op == 3:
            exit(0)
