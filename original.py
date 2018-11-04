import cv2
import numpy as np
import os


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
    """Reads training img path and returns two lists that relate
    an img with an integer (label for ech person to recognize)"""

    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # Lists for relations img-person
    faces = []
    labels = []

    # Go through each directory and read images within it
    for dir_name in dirs:

        # Ignoring folder that don't start with 's'
        if not dir_name.startswith("s"):
            continue

        # Getting levels from dir names
        label = int(dir_name.replace("s", ""))

        # Build path containing images for current subject
        # Sample: subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # Go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            # Build image path
            # Sample: image path = training-data/s1/1.jpg
            image_path = subject_dir_path + "/" + image_name

            # Read and display image
            image = cv2.imread(image_path)

            """# DEBUG
            img2 = image
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            det = cv2.CascadeClassifier('xml-files/haarcascades/haarcascade_frontalface_default.xml')
            face2 = det.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
            for (x, y, w, h) in face2:
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Training on image...", img2)
            # END DEBUG"""

            cv2.imshow("Training on image...", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Detect face, add faces and label to the lists
            face, rect = detectFace(image)
            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels


def drawRectangleText(img, rect, text):
    """Draw a rectangle with the given coordinates (rect) in the image"""
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    return img


def predict(test_img, recognizer):
    """Recognizes the person in the image and marks it with
    a rectangle and his/her name"""
    # Making a security copy of the img
    img = test_img

    # Detect face from the image
    face, rect = detectFace(img)

    # Predict the image using our face recognizer
    label = recognizer.predict(face)

    # Get name of label (first elem of tuple) returned by face recognizer
    label_text = subjects[label[0]]

    # Mark down the image with a rectangle and text
    img = drawRectangleText(img, rect, label_text)

    return img


if __name__ == "__main__":
    # No folder with 0, therefore, no person in list[0]
    subjects = ["", "Will Ferrel", "Bryan Cranston"]
    # Preparing training data: two lists (one of faces and
    # the other of labels) that relates a face with its label
    print("Preparing data...")
    faces, labels = prepareTrainingData("training-data")
    print("Data prepared")
    print("")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    # Creating our face recognizer and train it
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    print("Predicting images...")

    # load test images
    test_img1 = cv2.imread("test-data/test1.jpeg")
    test_img2 = cv2.imread("test-data/test3.jpeg")

    # perform a prediction
    predicted_img1 = predict(test_img1, face_recognizer)
    predicted_img2 = predict(test_img2, face_recognizer)
    print("Prediction complete")

    # display both images
    cv2.imshow(subjects[1], predicted_img1)
    cv2.waitKey(0)
    cv2.imshow(subjects[2], predicted_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # saving trained model
    # face_recognizer.save("model.yml")
