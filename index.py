import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
img_count_full=0
ret,color_img= cap.read()

while 1:
    ret,color_img= cap.read()
    # color_img=cv2.imread('/Users/anandkumar/Desktop/iris/my pic.jpeg')

    # convert to gray
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    # find face
    faces = face_cascade.detectMultiScale(gray_img)

    # draw face box
    for(x,y,w,h) in faces:
        cv2.rectangle(color_img,(x,y),(x+w,y+h),(255,0,0),2)
        face_gray = gray_img[y:y+h, x:x+w]
        face_color = color_img[y:y+h, x:x+w]

    # for eye box
    eyes = eye_cascade.detectMultiScale(gray_img)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(color_img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # show 
    cv2.imshow('img',color_img)
    key = cv2.waitKey(1)
    if key == 81 or key == 27 :
        cv2.destroyAllWindows()
        break


cap.release()
cv2.destroyAllWindows()