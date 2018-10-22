import cv2
# import sys # For argument pass (sys.argv[1])

if __name__ == "__main__":
    # Default sizes
    minFaceSize = 80
    maxFaceSize = 200

    faceCascade = cv2.CascadeClassifier('xml-files/haarcascades/haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        returnValue, frame = video_capture.read()
        if returnValue != 0:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(minFaceSize, minFaceSize),
                maxSize=(maxFaceSize, maxFaceSize)
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (0 + maxFaceSize, 0 + maxFaceSize), (255, 0, 0))  # Max size
            cv2.rectangle(frame, (maxFaceSize, 0), (maxFaceSize + minFaceSize, 0 + minFaceSize), (0, 0, 255))  # Min size

            # Display the resulting frame
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
