"""
The MIT License (MIT)

Copyright (c) 2015 Roger R. Labbe Jr

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.TION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import cv2
import cv2.cv as cv
import trackingalgos as ta
import camera as Camera
import math as m
from kalman2d import Kalman2D
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import filterpy.stats as stats

"""initialize old Kalman Filter"""
kalman = Kalman2D()

"""initialize new Kalman Filter"""
f1 = KalmanFilter(dim_x=4, dim_z=2)
dt = 1

"""initialize transition function"""
f1.F = np.array ([[1,dt,0,0],
				  [0,1,0,0],
				  [0,0,1,dt],
				  [0,0,0,1]])

"""no control in this filer"""
f1.u = 0

"""measurement matrix, is 2x4, assume they come in same units"""
f1.H = np.array([[1,0,0,0],[0,0,1,0]])

# print(f1.H)

"""measurement noise matrix, 2x2 because there are only 2 sensor inputs"""
f1.R = np.array([[5,0],
				[0,5]])

# print(f1.R)
"""process matrix, 1 for each state variable"""
f1.Q = np.eye(4)*0.1
# print(f1.Q)

"""set covariance matrix"""
f1.x = np.array([[0,0,0,0]]).T
# print(f1.x)
# print()
"""set values to large covariances"""
f1.P = np.eye(4)*500
# print(f1.P)

myCamera0 = Camera.Camera('../Media/forest.mp4', "cam0.avi")
#myCamera0 = Camera.Camera(0, "cam0.avi")
winName = "0", "1"

got_frame, frame = myCamera0.getFrame()
got_frame, frame = myCamera0.getFrame()
got_frame, frame = myCamera0.getFrame()

# frame = cv2.resize(frame,None,fx=1, fy=1, interpolation = cv2.INTER_NEAREST)

t0 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
avg_daw0 = np.float32(t0)
updated = False

x_estimate = frame.shape[0]
y_estimate = frame.shape[1]
x_prediction = frame.shape[0]
y_prediction = frame.shape[1]
prev_x = frame.shape[0]
prev_y = frame.shape[1]

height = frame.shape[0]
width = frame.shape[1]
# blank_image = np.ones((height,width), np.uint8)

xs, ys = [],[]
pxs, pys = [],[]

bs = ta.backgroundSubtractor()

while myCamera0.isOn():
	
	got_frame, frame = myCamera0.getFrame()
	
	if (not got_frame):
		break;

	t0 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	f0 = t0.copy()

	x, y = ta.track2(bs,f0,t0, avg_daw0)

	if((x != -1) | (y != -1)):
		ta.drawCross(frame, (x, y), 0, 0, 255)
		kalman.update(x,y)
		prev_x = x
		prev_y = y
		updated = True

		pos = [x,abs(y - height)]
		z = np.array([[pos[0]],[pos[1]]])

		f1.predict ()
		f1.update (z)

		xs.append (f1.x[0,0])
		ys.append (f1.x[2,0])
		pxs.append (pos[0]*1)
		pys.append(pos[1]*1)

		# plot covariance of x and y
		cov = np.array([ [f1.P[0,0], f1.P[2,0]],[f1.P[0,2], f1.P[2,2]] ])
		stats.plot_covariance_ellipse((f1.x[0,0], f1.x[2,0]), cov=cov, 
			facecolor='g', alpha=0.2)
	else:
		kalman.update(prev_x,prev_y)

	x_prediction,y_prediction = kalman.getPrediction()
	x_estimate,y_estimate = kalman.getEstimate()

	# cv2.ellipse(blank_image, (int(x_prediction),int(y_prediction)), (1,1), 0, 0, 180, (255,0,0), -1)
	ta.drawCross(frame, (int(x_prediction),int(y_prediction)), 255, 0, 0)
	ta.drawCross(frame, (int(x_estimate),int(y_estimate)), 0, 255, 0)

	cv2.imshow( winName[0], frame )
	# cv2.imshow( winName[1], blank_image)

	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break
	

cv2.destroyWindow("0")
cv2.destroyWindow("1")
myCamera0.off()
p1, = plt.plot (xs, ys, 'r--')
p2, = plt.plot (pxs, pys)
plt.legend([p1,p2], ['filter', 'measurement'], 2)
plt.show()
