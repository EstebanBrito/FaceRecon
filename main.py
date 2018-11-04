import cv2
import numpy as np
import os


def showProfileMenu():
    """Show a menu for profile management"""
    op2 = 1
    while True:
        while op2 < 1 or op2 > 4:
            print("ADMINISTRAR PERFILES")
            print("[ 1 ] --- Mostrar perfiles actuales")
            print("[ 2 ] --- Añadir perfil")
            print("[ 3 ] --- Eliminar perfil")
            print("[ 4 ] --- Regresar a menu principal")
            op2 = input("Ingresa el numero de tu eleccion: ")
            print("\n\n")

        if op2 == 1:
            showCurrentProfiles()
        elif op2 == 2:
            print("Adding profile")
            # Add profile
        elif op2 == 3:
            # Delete profile
            print("Deleting profile")
        elif op2 == 4:
            break


def showCurrentProfiles():
    if os.path.isfile("training-data/profiles.txt"):
        # Read profiles from file
        print("Leyendo perfiles")
        print()
    else:
        print("No existe perfil alguno")
        print()


if __name__ == "__main__":
    op = 1
    while True:
        while op < 1 or op > 3: # not 1 <= op <= 3:
            print("MENU DE RECON FACIAL:")
            print("[ 1 ] --- Iniciar Recon Facial")
            print("[ 2 ] --- Administrar perfiles faciales")
            print("[ 3 ] --- Salir")
            op = input("Ingresa el numero de tu eleccion: ")
            print("\n\n")

        if op == 1:
            if os.path.isfile("model.yml"):
                # Start facial recognition
                print("Iniciando recon")
                print()
                # startRecon()
            else:
                # Train the model and then start recog
                print("Training and recon")
                print()
                # prepareTrainingData()
                # startRecon()
        elif op == 2:
            showProfileMenu()
        elif op == 3:
            exit(0)
