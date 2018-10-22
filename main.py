import cv2
import numpy
import os


def convertToRGB(img):
    """Converts a BGR image to a RGB image"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def convertToGray(img):
    """Converts a BGR image to a gray scale image"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def displayImage(img, label="Test Image"):
    """Displays a window containing the image. Disappears on keypress"""
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # For displaying imgs ith matplotlib.pyplot
    # plt.imshow(gray_img1, cmap='gray')

# Detecting faces in images (img, scaleFactor(High->fast->lessMatch), minNeighbors(High->lessMatch->HighQ))
# Registering face frames coordinates within their images
    # for (x, y, w, h) in faces1:
        # cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

if __name__ == "__main__":
    # No folder with 0, therefore, no person in list[0]
    subjects = ["", "Ramiz Raja", "Elvis Presley"]
    pass
