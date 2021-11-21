from flask import Flask, render_template,url_for, request,redirect,send_file, Response
from werkzeug.utils import secure_filename
import cv2
import os
import tensorflow
from tensorflow.keras.models import load_model
import numpy as np



# mixer.init()
# sound = mixer.Sound('alarm.wav')


face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

lbl=['Close','Open']

model = load_model('cnnCat2.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

app=Flask(__name__)

# camera = cv2.VideoCapture('test.webm')

# video_stream = "0110c1b4-ce70-439a-954e-5d43fe09e9bf"
# video_stream = video_stream.encode()
# with open('file.webm', 'wb') as f_vid:
#     f_vid.write(base64.decodebytes(video_stream))

# with open('file.webm', 'rb') as f_vid:
#     video_stream = base64.b64decode(f_vid.read())

camera= cv2.VideoCapture('yo.webm')



def gen_frames():  
    global lbl, font, count, score, thicc, rpred, lpred, face, leye, reye, model, path

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            height,width = frame.shape[:2] 

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
            left_eye = leye.detectMultiScale(gray)
            right_eye =  reye.detectMultiScale(gray)
            cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
            for (x,y,w,h) in right_eye:
                r_eye=frame[y:y+h,x:x+w]
                count=count+1
                r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye,(24,24))
                r_eye= r_eye/255
                r_eye=  r_eye.reshape(24,24,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                rpred = model.predict_classes(r_eye)
                if(rpred[0]==1):
                    lbl='Open' 
                if(rpred[0]==0):
                    lbl='Closed'
                break
            for (x,y,w,h) in left_eye:
                l_eye=frame[y:y+h,x:x+w]
                count=count+1
                l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
                l_eye = cv2.resize(l_eye,(24,24))
                l_eye= l_eye/255
                l_eye=l_eye.reshape(24,24,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                lpred = model.predict_classes(l_eye)
                if(lpred[0]==1):
                    lbl='Open'   
                if(lpred[0]==0):
                    lbl='Closed'
                break
            if(rpred[0]==0 and lpred[0]==0):
                score=score+1
                cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            # if(rpred[0]==1 or lpred[0]==1):
            else:
                score=score-1
                cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
                
            if(score<0):
                score=0   
            cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            if(score>30):
                #person is feeling sleepy so we beep the alarm
                cv2.imwrite(os.path.join(path,'image.jpg'),frame)
                # try:
                #     sound.play()
                    
                # except:  # isplaying = False
                #     pass

                if(thicc<16):
                    thicc= thicc+2
                else:
                    thicc=thicc-2
                    if(thicc<2):
                        thicc=2
                cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 




            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
@app.route("/home")
def home():

    return render_template("home.html")


@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/meow')
def des():
    return render_template('des.html')



@app.route("/convert", methods=["GET", "POST"] )
def convert():
    global camera

    if request.method == "POST":

        if request.files:
            video = request.files["video"]
            if video.filename == "":
                 return render_template("ano.html")
            video.save('yo.webm')
            print(video.filename)

            camera= cv2.VideoCapture('yo.webm')
            return render_template("index.html")

    return render_template("ano.html")



if __name__=='__main__':
    app.run(debug=True)