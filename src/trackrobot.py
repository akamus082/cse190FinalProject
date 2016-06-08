import numpy as np
import cv2
import cv2.cv as cv
import trackingalgos as ta
import camera as Camera
import math as m
from kalman2d import Kalman2D

"""initialize old Kalman Filter"""
kalman = Kalman2D()

myCamera0 = Camera.Camera('../Media/video2.mp4', "video.avi")
#myCamera0 = Camera.Camera(0, "cam0.avi")
winName = "0", "1"

got_frame, frame = myCamera0.getFrame()
got_frame, frame = myCamera0.getFrame()
got_frame, frame = myCamera0.getFrame()

# frame = cv2.resize(frame,None,fx=1, fy=1, interpolation = cv2.INTER_NEAREST)

t0 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
avg_daw0 = np.float32(t0)
updated = False

height = frame.shape[0]
width = frame.shape[1]

bs = ta.backgroundSubtractor()
prev_x = frame.shape[0]
prev_y = frame.shape[1]

while myCamera0.isOn():
	
	got_frame, frame = myCamera0.getFrame()
	
	if (not got_frame):
		break;

	t0 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	f0 = t0.copy()

	x, y = ta.track(bs,f0,t0, avg_daw0)


	if((x != -1) | (y != -1)):
		ta.drawCross(frame, (x, y), 0, 0, 255)
		kalman.update(x,y)
		prev_x = x
		prev_y = y
		updated = True
	else:
		kalman.update(prev_x,prev_y)

	x_prediction,y_prediction = kalman.getPrediction()
	x_estimate,y_estimate = kalman.getEstimate()

	ta.drawCross(frame, (int(x_prediction),int(y_prediction)), 255, 0, 0)
	ta.drawCross(frame, (int(x_estimate),int(y_estimate)), 0, 255, 0)

	cv2.imshow( winName[0], frame )
	# cv2.imshow( winName[1], img)

	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break
	

# cv2.destroyWindow("0")
cv2.destroyWindow("1")
myCamera0.off()
