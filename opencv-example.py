import cv2

image = cv2.imread('image.jpg')

while True:

    cv2.imshow('Desktop', image)

    key=cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()