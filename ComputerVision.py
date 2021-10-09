import cv2
from imutils.convenience import rotate
import pytesseract
import imutils
from imutils import contours
import numpy as np
import os
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    imgContour = img.copy()  # Change - make a copy of the image to return

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

                index = i  # Also save index to contour

    if index is not None:  # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

    return biggest, imgContour  # Change - also return drawn image


def PlateReg(value):
    file = '' + value + '.jpeg'  # 8, 14, 22 + 0(?), 4, 18(?),

    # cv2.IMREAD_COLOR
    image = cv2.imread(('images/' + file), cv2.IMREAD_COLOR) #считывание исходника
    image = imutils.resize(image, width=900) #изменение ширины (эксперемантально подобрал 900, показывает лучший результат)
    # height, width, _ = image.shape

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    # gray = cv2.bilateralFilter(gray, 11, 17, 20)  # Blur to reduce noise
    # gray = cv2.GaussianBlur(gray, (1, 1), 0)
    # edged = cv2.Canny(gray, 30, 255)  # Perform Edge detection

    kernel = np.ones((3, 3))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #перевод в чб
    gray = cv2.GaussianBlur(gray, (1, 5), 1) #размытие по Гауссу
    gray = cv2.bilateralFilter(gray, 11, 17, 20)  #доп размытие

    # edged = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edged = cv2.dilate(gray, kernel, iterations=1) #
    # edged = cv2.erode(edged, kernel, iterations=0)
    edged = cv2.Canny(edged, 0, 255) #трассировка в границы Канни 

    gray = cv2.threshold(gray, 114, 255, cv2.THRESH_BINARY)[ 
        1]  #трассировка в бинарное изображение (114=64+32+12+6 эксперементально)

    # gray = cv2.threshold(gray, 114, 255, cv2.THRESH_OTSU)[
    #     1]  # 64+32+12+6 эксперементально

    biggest, imgContour = getContours(gray)  # Change

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(imgContour.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 9, True)  # 36
        # approx = cv2.approxPolyDP(c, 0.036 * peri, closed=1) #36
        # print(cnts)
        if len(approx) == 4:
            screenCnt = approx

            break

    #!!
    # print("biggest = " + str(biggest))
    # print("screenCnt = " + str(screenCnt))

    im_src = image
    pts_src = approx #массив координат пикселей автомобильного номера в исходном изображении

    im_dst = cv2.imread('orig.jpeg') #считывание изображения

    pts_dst = np.array([ [0, 0],[0, 458],[98, 458],[98, 0]]) #массив координат пикселей трафаретного изображения

    h, status = cv2.findHomography(pts_src, pts_dst) #совмещяем дваи изображения, находим гомографию

    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[0], im_dst.shape[1])) #растягиваем изображение

    flip= cv2.flip(im_out,1) #отзеркаливание

    
    rotated = cv2.rotate(flip, cv2.ROTATE_90_CLOCKWISE) #поворот на 90 градусов
    rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) #преобразование варпленного изображения в чб

    # 
    
    
    #rotated = cv2.GaussianBlur(rotated, (1, 5), 1)
    #rotated = cv2.bilateralFilter(rotated, 11, 17, 20)  # Blur to reduce noise

    #rotated = cv2.threshold(rotated, 170, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    #rotated = cv2.Canny(rotated, 0, 255)

    #rotated = cv2.threshold(rotated, 250, 255, cv2.THRESH_OTSU)[1]  # 64+32+12+6 эксперементально
    # rotated = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 1)
    #rotated = cv2.equalizeHist(rotated)
    #64+32+12+6, 255, cv2.THRESH_BINARY)[1]  # 64+32+12+6 эксперементально
    # rotated = cv2.dilate(rotated, None, iterations=2)
    # rotated = cv2.erode(rotated, None, iterations=2)
    # imgContour
    #print(approx)

    #!

    if screenCnt is None:
        detected = 0
        print("No contour detected")
    else:
        detected = 1

    if detected == 1:

        #cv2.drawContours(imgContour, [screenCnt], -1, (255, 0, 0), 1)
        cv2.drawContours(image, [screenCnt], -1, (200, 70, 255), 2)

    # Маска прозрачности
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(image, image, mask=mask)

    # Обрезаем
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = imgContour[topx:bottomx + 1, topy:bottomy + 1]  # ! gray
    Cropped = imutils.resize(Cropped, width=1000)
    #Cropped = imutils.resize(Cropped, width=520, height=112)

    cv2.imwrite('Cropped.jpg', Cropped)
    cv2.imwrite('rotated.jpg', rotated)

    cv2.imshow("Warped Source Image", rotated)
    cv2.imshow('Edged', imgContour)
    cv2.imshow('Stock', image)
    #cv2.imshow('Cropped', Cropped)
    #cv2.imshow("Warped Source Image", im_out)
    # for c in cnts:
    #     area = cv2.contourArea(c)
    #     x,y,w,h = cv2.boundingRect(c)
    #     if area >100:

    result = pytesseract.image_to_string(
        'rotated.jpg', lang='rus+eng', config='-c tessedit_char_whitelist=ABCEHIKMOPTYX0123456789') #определение символов на изображении

    print(result)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # os.remove("Cropped.jpg")

    return result


PlateReg(input())
