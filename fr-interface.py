# THIS PACKAGE CONTAINS GENERIC METHODS FOR EASIER MEDIA
# MANIPULATION, FACE DETECTION/RECOGNITION AND DATA OUTPUT
# Key package for computer vision
import cv2
# For maths operations
import numpy
# For interactions with the OS
import os
import sys
# Other useful packages
import time


def convertToRGB(img):
    """Converts a BGR image to a RGB image
    Useful for displaying images for matplotlib.pyplot"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def convertToGray(img):
    """Converts a BGR image to a gray scale image.
    Useful for better face detection performance"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def displayImage(img, label="Test Image"):
    """Displays a window containing the image. Disappears on keypress"""
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # For displaying imgs ith matplotlib.pyplot
    # plt.imshow(gray_img1, cmap='gray')


def detectFaces(img, classifier, qualityLevel=1):
    """Returns img with the faces detected in it. Quality level
    can be changed for better performance (the higher the integer,
    the higher the quality and slower the process)"""
    # scaleFactor(High->fast->lessMatch), minNeighbors(High->lessMatch->HighQ))
    scale = 0
    neigh = 0
    if qualityLevel==1:
        scale = 1.5
        neigh = 5
    elif qualityLevel == 2:
        scale = 1.3
        neigh = 7
    elif qualityLevel == 3:
        scale = 1.1
        neigh = 9

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(
        gray_img,
        scaleFactor=scale,
        minNeighboors=neigh,
        minSize=(80, 80),
        maxSize=(200, 200)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


def detectFacesDeveloper(img, classifier):
    """Detects faces with information for debugging/developing"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighboors=5,
        minSize=(80, 80),
        maxSize=(200, 200)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(img, (80, 0), (160, 80), (0, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (200, 200), (255, 0, 0), 2)

    return img


def other():
    pass
