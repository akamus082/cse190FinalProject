import numpy as np
import cv2
import time

class Camera:
	
	def __init__(self, deviceNumber, filename):
		self.devNum = deviceNumber
		self.filename = filename
		self.filecount = 0
		self.got_frame = False
		self.cap = cv2.VideoCapture(self.devNum)
		self.cap.set(5, 30)
		self.cap.set(3, 640)
		self.cap.set(4, 480)


	def on(self):
		# turn the camera on
		#self.inUse = True
		self.cap.open(self.devNum)

	def off(self):
		# turn the camera off
		#self.inUse = False
		self.cap.release()

	def isOn(self):
		# return true if the camera is on
		return self.cap.isOpened()
		
	def isOff(self):
		# return true if the camera is off
		return not self.cap.isOpened()

	def getFrame(self):
		# get the next video frame
		self.got_frame, self.frame = self.cap.read()
		return self.got_frame, self.frame

	def getFrameLowRes(self):
		# get the next video frame
		self.got_frame, self.frame = self.cap.read()
		lowres = cv2.resize(self.frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
		return self.got_frame, lowres

	# This function is still in production.
	def getAndWriteFrame(self):
		# get the next video frame
		self.got_frame, self.frame = self.cap.read()
		# filename = "pic_" + str(self.devNum)+ "_%04d" % self.filecount
		# f = open(filename, 'w')
		# f.write(self.frame.tostring())
		# f.close()
		# # Update the file count.
		# self.filecount += 1
		# # <<<< INCLUDE OPTION TO LOWER THE RES BEFORE RETURNING ?
		lowres = cv2.resize(self.frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)
		return self.got_frame, lowres 
	
	def getName(self):
		return str(self.filename)

	def getDev(self):
		return deviceNumber

	def gotFrame(sef):
		return self.got_frame



# myCamera = Camera(0, "mytestfile.avi")

# myCamera.on()
# time.sleep(2)
# print (myCamera.isOn() == True)
# print (myCamera.isOff() == False)
# time.sleep(2)
# myCamera.off()
# print myCamera.isOn()
# print myCamera.isOff()

# print "goodbye"

