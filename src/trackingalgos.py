import cv2
import numpy as np
MIN_BLOB_SIZE_ROBOT = 5
MIN_BLOB_SIZE = 50
kernel = np.ones((5,5),np.uint8)

class backgroundSubtractor(object):
    def __init__(self):
        self.bg_subtractor = cv2.BackgroundSubtractorMOG(history=200, 
            nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
def drawCross(img, center, r, g, b):
	'''
	Draws a cross a the specified X,Y coordinates with color RGB
	'''

	d = 5
	t = 2

	color = (r, g, b)

	ctrx = center[0]
	ctry = center[1]

	cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, cv2.CV_AA)
	cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, cv2.CV_AA)


def getCentroid(contour_in):
    moments = cv2.moments(contour_in)
    x = int(moments['m10'] / moments['m00'])
    y = int(moments['m01'] / moments['m00'])
    return [x, y]

def track(bs,img_copy,img, avg):
	x = -1
	y = -1

	img_copy = cv2.GaussianBlur(img_copy,(5,5),0)
	cv2.accumulateWeighted(img_copy,avg,0.4)
	res = cv2.convertScaleAbs(avg)

	res = bs.bg_subtractor.apply(res, None, 0.05)

	gradient = cv2.morphologyEx(res, cv2.MORPH_GRADIENT, kernel)

	processed_img = cv2.GaussianBlur(gradient,(5,5),0)

	_,threshold_img = cv2.threshold( processed_img, 20, 255, cv2.THRESH_BINARY )

	if np.count_nonzero(threshold_img) > 5:

		contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, 
			cv2.CHAIN_APPROX_SIMPLE)

		# totally not from stack overflow
		areas = [cv2.contourArea(c) for c in contours]
		# max_index  = np.argmax(areas)
		max_index = np.argmin(areas)
		# Make sure it's big enough
		if cv2.contourArea(contours[max_index]) >= MIN_BLOB_SIZE_ROBOT:
			# img_out = np.zeros(img_thresh.shape).astype(np.uint8)
			cv2.drawContours(img, contours, max_index, (255, 255, 255), -1)
			x, y = getCentroid(contours[max_index])

	return x, y

def track2(bs,img_copy,img, avg):
	x = -1
	y = -1

	img_copy = cv2.GaussianBlur(img_copy,(5,5),0)
	cv2.accumulateWeighted(img_copy,avg,0.4)
	res = cv2.convertScaleAbs(avg)
	res = cv2.absdiff(img, res)
	_,processed_img = cv2.threshold( res, 7, 255, cv2.THRESH_BINARY )
	processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
	_,processed_img = cv2.threshold( processed_img, 240, 255, cv2.THRESH_BINARY )

	processed_img = bs.bg_subtractor.apply(processed_img, None, 0.05)
	
	# img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
	
	if np.count_nonzero(processed_img) > 5:
		# Get the largest contour
		contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, 
			cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)

		# Make sure it's big enough
		if cv2.contourArea(contours[max_index]) >= MIN_BLOB_SIZE:
			cv2.drawContours(img, contours, max_index, (255, 255, 255), -1)
			x, y = getCentroid(contours[max_index])

	return x, y