from time import sleep
import numpy as np
import cv2
import os



def part1():
    bg = cv2.imread('exercisefiles/Test016/001.tif')


    cv2.imshow('Background',bg)
    cv2.waitKey(0)

    
    images = load_images_from_folder('exercisefiles/Test016/')
    
    
    for i in images:
        diff = cv2.absdiff(bg, i)
        cv2.imshow('Diff', diff)
        cv2.waitKey(0)




def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def part2():

    images = load_images_from_folder('exercisefiles/Test016/')

    image_array = np.array(images)

    bg = np.median(image_array, axis=0).astype(np.uint8)


    cv2.imshow('Background',bg)
    cv2.waitKey(0)


    for i in images:
        diff = cv2.absdiff(bg, i)
        cv2.imshow('Diff', diff)
        cv2.waitKey(0)



def part3():
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    images = load_images_from_folder('exercisefiles/Test016/')

    for i in images:
        fg_mask = bg_subtractor.apply(i)

        cv2.imshow('Original', i)
        cv2.imshow('Foreground mask', fg_mask)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    part3()