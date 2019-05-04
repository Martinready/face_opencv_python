import cv2,time
import numpy as np
from gtts import gTTS
import sqlite3#SQLite資料庫
import os, sys
from os import listdir#os 尋找資料夾
from os.path import isfile, isdir, join#os walk
import paho.mqtt.client as mqtt#MQTT接收端
import paho.mqtt.publish as publish#MQTT發送端
import threading#多執行緒
import pyodbc#連線SQL
from PIL import Image
import urllib.request
import requests





#建資料庫
#辨識身分
#有會員；登入
#無會員:註冊會員
#拍訓練用相片
#訓練
server = 'magicmirror1.database.windows.net'
database = 'MagicMirror'
username = 'azureuser'
password = 'Azure1234567'
connectionstring = 'driver={FreeTDS};server=' + server + ';port=1433;database=' + database + ';uid=' + username + ';pwd=' + password + ';TDS_Version=7.2'
conn2 = pyodbc.connect(connectionstring)
cursor2 = conn2.cursor()
print("x")
#多執行緒
def sub():
	time.sleep(1)
	os.system("python3 /home/pi/a/w.py")
	print("sub")
t = threading.Thread(target = sub)
# 執行該子執行緒
t.start()

fname2 = "./database.db"#找資料庫
if not os.path.isfile(fname2):
	conn = sqlite3.connect('database.db')#連線
	c = conn.cursor()
	sql = """
	DROP TABLE IF EXISTS users;
	CREATE TABLE users (
			   id integer unique primary key autoincrement,
			   name text
	);
	"""
	c.executescript(sql)
	conn.commit()#資料庫提交
	conn.close()#連線斷開




path2 = "/home/pi/a/images"

conn = sqlite3.connect('database.db')
c = conn.cursor()
fname = "recognizer/trainingData.yml"#訓練檔
#是否有訓練檔
if not os.path.isfile(fname):
	print("Please train the data first")
	exit(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#辨識用檔案
cap = cv2.VideoCapture(0)#攝影機號碼
recognizer = cv2.face.LBPHFaceRecognizer_create()#建立訓練
recognizer.read(fname)#載入訓練
ESC =27
#迴圈
aaa = ""
ab=""
###
nnn=0#計算辨識次數
num_dirs = 0#計算資料夾
#辨識

while True:
	ret, img = cap.read()
	#img = cv2.resize(img, (960, 720))
	img = cv2.flip(img, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
		face_img = cv2.resize(gray[y: y + h, x: x + w], (200, 200))
		ids,conf = recognizer.predict(face_img)
		c.execute("select name from users where id = (?);", (ids,))#比對id
		result = c.fetchall()
		name = result[0][0]
		print(conf)
		if conf < 55:
###########有註冊
			#os.system("sudo python3 0408_web.py")
			print("OK")
			cursor2.execute("SELECT * FROM userinfo WHERE imgfile = ?", name) 
			row2 = cursor2.fetchone() 
			while row2:
				print (row2.name)
				if row2.name==None:
					print("x")
					hello=row2.imgfile
				else:
					hello=row2.name
				row2 = cursor2.fetchone()
			url = 'http://tts.baidu.com/text2audio?idx=1&tex=你好，'+hello+'，歡迎回家&cuid=baidu_speech_demo&cod=2&lan=zh&ctp=1&pdt=1&spd=5&per=4&vol=5&pit=5&fbclid=IwAR2QmepfO74oJRgNSNjBB1vniZthLInjLo4vyXbVNDxNG_snj1jQEyziW1M'
			r = requests.get(url, allow_redirects=True)
			open('hello.mp3', 'wb').write(r.content)
			print(name)#顯示id
			os.system("mpg321 -q /home/pi/ok.mp3")#撥放音樂
			os.system("mpg321 -q /home/pi/a/hello.mp3")
			cv2.putText(img, name, (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)#畫框框
			time.sleep(3)
			cv2.destroyAllWindows()
			aaa = "000"#跳出迴圈用
##########websocket_pub
			#publish.single("Magic",name, hostname="m16.cloudmqtt.com",port=11536, auth={"username": "dkwzubzf", "password": "fdZGJ3gAy2mL"})
			publish.single("Magic",name, hostname="berry")
#########################
			print("Magic")#debug
			break
		elif 50< conf:
###########沒註冊
			cv2.putText(img, 'No Match', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)#畫框框
			nnn+=1#辨識一次
			print(name)
			print(nnn)
			#辨識幾次
			if nnn==50:
#############開始註冊
				print("註冊會員")
				ab="000"#跳出迴圈
				for root,dirs,files in os.walk(path2):    #遍歷統計資料夾
						for name in dirs:
								num_dirs += 1
				print (num_dirs)
				number=num_dirs+1
				begin="h"+str(number)#使用者帳號(預設為資料夾名稱)
				if not os.path.exists('./dataset'):#建立dataset資料夾
					os.makedirs('./dataset')
				uname = begin#資料庫變數
				url = 'http://tts.baidu.com/text2audio?idx=1&tex=你好，目前還沒有你的資料&cuid=baidu_speech_demo&cod=2&lan=zh&ctp=1&pdt=1&spd=5&per=4&vol=5&pit=5&fbclid=IwAR2QmepfO74oJRgNSNjBB1vniZthLInjLo4vyXbVNDxNG_snj1jQEyziW1M'
				r = requests.get(url, allow_redirects=True)
				open('who.mp3', 'wb').write(r.content)
				os.system("mpg321 -q /home/pi/a/who.mp3")
				c.execute('INSERT INTO users (name) VALUES (?)', (uname,))#插入資料
				uid = c.lastrowid
				# abc=str(random.randint(10,100))
				# xx=random.choice('abcdefg')
				# account2=xx+abc
				#print(account2)
				#新增一筆UserName: 王阿壹、UserPassword: 0000的資料到UserInfo
				cursor2.execute("INSERT INTO userinfo (account, pwd,imgfile) VALUES (?, ?,?)", uname,"123",uname)
				conn2.commit()
				conn2.close()
				print("new ok")
				sampleNum = 0#拍照次數
				path3 = "/home/pi/a/images/"+begin
				if not os.path.exists(path3):
					os.makedirs(path3)
					while True:
						ret, img = cap.read()
						#img = cv2.resize(img, (960, 720))#視窗大小調整
						img = cv2.flip(img, 1)#視窗翻轉
						gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
						faces = face_cascade.detectMultiScale(gray, 1.3, 5)
						for (x,y,w,h) in faces:
							sampleNum = sampleNum+1
							cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])#在資料夾存放圖片
							cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
							print("1")
							cv2.waitKey(100)
						cv2.imshow('Face Recognizer',img)
						cv2.waitKey(1);
						#次數大於幾次
						if sampleNum > 41 :
							break
					cap.release()
					conn.commit()
					conn.close()
				########建立訓練模型
				recognizer = cv2.face.LBPHFaceRecognizer_create()
				path = 'dataset'#路徑
				if not os.path.exists('./recognizer'):#建立資料夾
					os.makedirs('./recognizer')
					#建立
				def getImagesWithID(path):
					imagePaths = [os.path.join(path,f) for f in os.listdir(path)]#存取圖片
					faces = []
					IDs = []
					for imagePath in imagePaths:
						faceImg = Image.open(imagePath).convert('L')
						faceNp = np.array(faceImg,'uint8')
						ID = int(os.path.split(imagePath)[-1].split('.')[1])
						faces.append(faceNp)
						IDs.append(ID)
						cv2.waitKey(10)
					return np.array(IDs), faces
				Ids, faces = getImagesWithID(path)
				recognizer.train(faces,Ids)
				recognizer.save('recognizer/trainingData.yml')
				#os.system("sudo python3 0408_web.py")
				publish.single("Magic",uname, hostname="berry")
				print('trining done')
				cv2.destroyAllWindows()
			break
		else:
			cv2.putText(img, 'No Match', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)#畫框框
			print(name)
	cv2.moveWindow('Face Recognizer', 500,230)#固定視窗位置
	cv2.imshow('Face Recognizer',img)#視窗顯示
###跳出迴圈
	if aaa == "000"or ab=="000":
		break
	if cv2.waitKey(1) == ESC:
		cv2.destroyAllWindows()
		break
cap.release()

