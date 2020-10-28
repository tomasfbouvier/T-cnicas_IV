import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

DIGITS_LOOKUP = {
  (1, 1, 1, 0, 1, 1, 1): 0,
  (0, 0, 1, 0, 0, 1, 0): 1,
  (1, 0, 1, 1, 1, 1, 0): 2,
  (1, 0, 1, 1, 0, 1, 1): 3,
  (0, 1, 1, 1, 0, 1, 0): 4,
  (1, 1, 0, 1, 0, 1, 1): 5,
  (1, 1, 0, 1, 1, 1, 1): 6,
  (1, 0, 1, 0, 0, 1, 0): 7,
  (1, 1, 1, 1, 1, 1, 1): 8,
  (1, 1, 1, 1, 0, 1, 1): 9
}

# Leer Vídeo
cap = cv2.VideoCapture('VID_20200226_131559.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
n=0
# Read until video is completed
while(cap.isOpened()):
  ret, frame = cap.read()

  # Capture frame-by-frame
  if n%30==0:
    frame=np.rot90(frame)
    frame=np.rot90(frame)
    frame=np.rot90(frame)
    image=cv2.resize(frame,(640,480))
    image=image[180:280,0:640]
  #  image=cv2.resize(image,(640,480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    #ya cambiamos el formato de la imagen, vamos a hacer cosas que no sabemos

    #thresh = cv2.threshold(edged, 0, 255,
     # cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

 #   cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
  #    cv2.CHAIN_APPROX_SIMPLE)
   # cnts = imutils.grab_contours(cnts)
    #digitCnts = []
    # loop over the digit area candidates
 #   for c in cnts:
      # compute the bounding box of the contour
  #    (x, y, w, h) = cv2.boundingRect(c)
      # if the contour is sufficiently large, it must be a digit
   #   if w >= 15 and (h >= 30 and h <= 40):
    #    digitCnts.append(c)
    
    if ret == True:

    		# Display the resulting frame
      cv2.imshow('Frame',image)
      #cv2.imshow('Frame',cnts)
      
      if cv2.waitKey(0) & 0xFF == ord('q'):
       cv2.destroyAllWindows()
      if cv2.waitKey(400)==27:
       print("video parado al pulsar esc")
       break
    	
  # Break the loop
    else: 
      break
  n+=1
  
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()