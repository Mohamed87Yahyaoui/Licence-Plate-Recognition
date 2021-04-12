import cv2
import pytesseract as pyt
import numpy as np

# Global variables
widthWindow = 640
heightWindow = 500
brightnessWindow = 100
count = 0

# Recognition tools
pyt.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
plateNumberCascade = cv2.CascadeClassifier("venv/Lib/site-packages/cv2/data/haarcascade_russian_plate_number.xml")

# Open mobile webcam
cap1 = cv2.VideoCapture(1)

# Set camera size
cap1.set(3, widthWindow)
cap1.set(4, heightWindow)
cap1.set(10, brightnessWindow)

# Body
while True:
    success, img = cap1.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Number = plateNumberCascade.detectMultiScale(imgGray, 1.2, 1)
    for x, y, w, h in Number:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        imgWanted = img[y:y+h, x:x+w]
        imgWantedRGB = cv2.cvtColor(imgWanted, cv2.COLOR_BGR2RGB)
        num = pyt.image_to_string(imgWantedRGB, config='--psm 6')
        print(num)
        cv2.imshow("Number Plate", imgWanted)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        cv2.imwrite("Images/ScannResult/NoPlate"+str(count)+".png", imgWanted)
        count = count + 1
        cv2.rectangle(img, (0, 200), (640, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Number successfully saved", (50, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv2.imshow("Image", img)
        cv2.waitKey(5000)
        break