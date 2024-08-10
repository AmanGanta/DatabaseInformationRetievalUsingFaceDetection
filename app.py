from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import sqlite3
import cv2
import cvlib as cv
from flask import Flask, request, render_template
l=['Aman', 'Dhoni', 'Dua lipa', 'Ram Charan', 'biden', 'dq', 'krithi shetty', 'nani', 'sai pallavi','Samantha' 'sree leela', 'virat']
model1=load_model(r"D:\projects\Main projects gec\face recognition\deep.h5")
model2=load_model(r"D:\projects\Main projects gec\face recognition\deep19.h5")
app = Flask(__name__)
@app.route('/home', methods=['GET'])
def index():
    return render_template('home.html')
@app.route('/process_login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        q1=request.form.get("username")
        q2=request.form.get("password")
        conn = sqlite3.connect('verify_data.db')
        cursor = conn.cursor()
        cursor.execute('''
                       SELECT  f.pickle_data
                       FROM users f
                       JOIN users u ON f.id = u.id
                       WHERE u.username = ? AND u.password = ?
                       ''', (q1, q2))
        result = cursor.fetchone()
        if result:
            file_data = result[0]
            
            h5_file_path =r"D:\projects\Main projects gec\face recognition" + q1 + '.h5'
            print("*"*500)
            print(h5_file_path)
            with open(h5_file_path, 'wb') as file:
                file.write(file_data)

            model = load_model(h5_file_path)
            
            webcam = cv2.VideoCapture(0)
            while webcam.isOpened():
               status, frame = webcam.read() 
               face, confidence = cv.detect_face(frame)
               if(len(face)==0):
                   return render_template('result.html',data="please place your camera properly and try again")
               for idx,f in enumerate(face):
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                    face_crop = np.copy(frame[startY:endY,startX:endX])
                    if(face_crop is None):
                        continue
                    face_crop = cv2.resize(face_crop, (224,224))
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)
                    p = model.predict(face_crop)
                    a=model1.predict(face_crop)
                    z=model2.predict(face_crop)
               break
            webcam.release()
            cv2.destroyAllWindows()
            #print(a)
            print(z)
            k=np.argmax(a[0],axis=0)
            k1=np.argmax(z[0],axis=0)
            print(k1)
            j=max(a[0])
            j1=max(z[0])
            if(p[0][0]):
                cursor.execute('''
                               SELECT  f.details
                               FROM users f
                               JOIN users u ON f.id = u.id
                               WHERE u.username = ? AND u.password = ?
                               ''', (q1, q2))
                result = cursor.fetchone()
                if(j>0.80 or j1>0.80):    
                    if(l[k] in q1 or l[k1] in q1):
                        return render_template('result.html',data=result[0])
                    else:
                       return render_template('result.html',data="Invalid user") 
                else:
                    return render_template('result.html',data=result[0])
            else:
                return render_template('result.html',data="Invalid user")        
        else:
           return render_template('result.html',data='invalid Information')

if __name__ == '__main__':
    app.run(debug=True)
    