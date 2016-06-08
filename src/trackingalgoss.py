import cv2
import numpy as np
import CameraTrapCV as CTCV
ctcv = CTCV.CameraTrapCV()
MIN_BLOB_SIZE = 50
kernel = np.ones((5,5),np.uint8)

def getEstimate(x,y, f,t, avg):

	f = cv2.GaussianBlur(f,(5,5),0)
	cv2.accumulateWeighted(f,avg,0.4)
	res = cv2.convertScaleAbs(avg)
	
	res2 = cv2.absdiff(t, res.copy())
	ret,img_grey2 = cv2.threshold( res2, 7, 255, cv2.THRESH_BINARY )
	img_grey2 = cv2.GaussianBlur(img_grey2,(5,5),0)
	ret2,img_grey2 = cv2.threshold( img_grey2, 240, 255, cv2.THRESH_BINARY )

	img_thresh = ctcv.bg_subtractor.apply(img_grey2, None, 0.05)
	# erosion = cv2.erode(img_thresh,kernel,iterations = 1)
	# img_thresh = cv2.dilate(erosion,kernel,iterations = 1)
	img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

	if np.count_nonzero(img_thresh) > 1:
		# Get the largest contour
		contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# totally not from stack overflow
		areas = [cv2.contourArea(c) for c in contours]
		i_max  = np.argmax(areas)
		max_index = ctcv.getLargestContourIndex(img_thresh)

		# Make sure it's big enough
		if cv2.contourArea(contours[max_index]) >= MIN_BLOB_SIZE:
			img_out = np.zeros(img_thresh.shape).astype(np.uint8)
			cv2.drawContours(t, contours, max_index, (255, 255, 255), -1)
			x, y = ctcv.getCentroid(contours[max_index])

	return x, y

def diffaccWeight(f,t, avg):
	x = -1
	y = -1

	f = cv2.GaussianBlur(f,(5,5),0)
	cv2.accumulateWeighted(f,avg,0.4)
	res = cv2.convertScaleAbs(avg)
	res2 = cv2.absdiff(t, res.copy())
	ret,img_grey2 = cv2.threshold( res2, 7, 255, cv2.THRESH_BINARY )
	img_grey2 = cv2.GaussianBlur(img_grey2,(5,5),0)
	ret2,img_grey2 = cv2.threshold( img_grey2, 240, 255, cv2.THRESH_BINARY )

	img_thresh = ctcv.bg_subtractor.apply(img_grey2, None, 0.05)
	
	# img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
	
	if np.count_nonzero(img_thresh) > 5:
		# Get the largest contour
		contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# totally not from stack overflow
		areas = [cv2.contourArea(c) for c in contours]
		i_max  = np.argmax(areas)
		max_index = ctcv.getLargestContourIndex(img_thresh)

		# Make sure it's big enough
		if cv2.contourArea(contours[max_index]) >= MIN_BLOB_SIZE:
			# img_out = np.zeros(img_thresh.shape).astype(np.uint8)
			cv2.drawContours(t, contours, max_index, (255, 255, 255), -1)
			x, y = ctcv.getCentroid(contours[max_index])

	return t, x, y


def runningAvg(frame, running_average_in_display, avg):
	x_pos = 0
	y_pos = 0
	display_image = frame.copy()

	#smooth image
	blur = cv2.GaussianBlur(display_image,(5,5),0)

	#calculate running avg
	cv2.accumulateWeighted(blur,avg,0.5)
	res = cv2.convertScaleAbs(avg)
	cv2.convertScaleAbs(avg,running_average_in_display, 1.0, 0.0)

	#get the difference between avg and image
	difference = cv2.absdiff(display_image, running_average_in_display)

	#convert image to grayscale
	img_grey = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)

	#compute threshold
	ret,img_grey = cv2.threshold( img_grey, 2, 255, cv2.THRESH_BINARY )

	#smooth and threshold again to eliminate sparkles
	img_grey = cv2.GaussianBlur(img_grey,(5,5),0)
	ret,img_grey = cv2.threshold( img_grey, 240, 255, cv2.THRESH_BINARY )

	grey_image_as_array = np.asarray(  img_grey  )
	non_black_coords_array = np.where( grey_image_as_array > 3 )
	non_black_coords_array = zip( non_black_coords_array[1], non_black_coords_array[0] )

	bounding_box_list = []

	contour,hier = cv2.findContours(img_grey,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

	areas = [cv2.contourArea(c) for c in contour]
	max_index = np.argmax(areas)
	cnt=contour[max_index]
	if (cv2.contourArea(contour[max_index]) > 5000):
		polygon_points = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
		bounding_rect = cv2.boundingRect( cnt )
		point1 = ( bounding_rect[0], bounding_rect[1] )
		point2 = ( bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3] )
		bounding_box_list.append( ( point1, point2 ) )
		cv2.fillPoly( img_grey, [ polygon_points ], (255,255,255) )
		x,y,w,h = cv2.boundingRect(cnt)
		x_pos, y_pos = ctcv.getCentroid(contour[max_index])
		cv2.circle(display_image, (x_pos, y_pos), 3, (0,0,0), -1)
		cv2.rectangle(display_image,(x,y),(x+w,y+h),(0,255,0),2)
		#cv2.imshow( "2", display_image )
		#return display_image, x_pos, y_pos
		return display_image, x_pos, y_pos
	
	return frame, x_pos, y_pos
