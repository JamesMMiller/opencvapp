import cv2
import numpy as np
import imutils
from kivy.storage.jsonstore import JsonStore

class ObjectTracker():
	def __init__(self,image,hsv_thresholds,objectsize,mtx,dist):
		self.image = image
		self.hsv_thresholds = hsv_thresholds
		self.objectsize = objectsize
		self.mtx = mtx
		self.dist = dist

	def calculate_distance(self, width_of_object, focalLength, detectedwidth):

		return ((float(width_of_object) * float(focalLength)) / float(detectedwidth))
	def get_mask(self):
		hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

		hsv_t = self.hsv_thresholds

		lower_threshold = np.array([hsv_t[0],hsv_t[1],hsv_t[2]])
		upper_threshold = np.array([hsv_t[3],hsv_t[4],hsv_t[5]])

		mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
		#mask = cv2.GaussianBlur(mask, (11, 11), 0)

		return mask
	def get_mask_res(self):
		mask = self.get_mask()
		mask_res = cv2.bitwise_and(self.image,self.image, mask= mask)
		return mask_res

	def calculate_circle(self):
		mask = self.get_mask()

		#now find contours in order to approximate a circle
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		center = None

		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			try:
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			except ZeroDivisionError:
				#handle division error
				center = ((0),(0))
			return x,y,radius,center
		return None, None, None, None

	def calculate_distance(self):
		x,y,radius,center = self.calculate_circle()
		focalLength = self.mtx[0, 0]
		width_of_object = self.objectsize * 10

		if radius != None:
			detectedwidth = radius * 2

			return ((float(width_of_object) * float(focalLength)) / float(detectedwidth))
		else:
			return None
	
	def draw_circle(self):
		x,y,radius,center = self.calculate_circle()

		F = self.mtx[0, 0]
		W = self.objectsize*10
			# only proceed if the radius meets a minimum size
		if radius != None:
			distance = self.calculate_distance() 
			if radius > 10:# draw the circle and centroid on the result,
				cv2.circle(self.image, (int(x), int(y)), int(radius),(0, 255, 255), 2)
				cv2.circle(self.image, center, 5, (0, 0, 255), -1)
			cv2.putText(self.image, "%.2fM" % (distance/1000),
				(self.image.shape[1] - 200, self.image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
				2.0, (0, 255, 0), 3)
		return self.image
		
	def undistort_point(self):
		x,y,radius,center = self.calculate_circle()
		z = self.calculate_distance()
		assert z != None


		f_x = self.mtx[0, 0]
		f_y = self.mtx[1, 1]
		c_x = self.mtx[0, 2]
		c_y = self.mtx[1, 2]

		#points = [x,y]
		points = np.array([x, y]).reshape(1,2)
		points = np.ascontiguousarray(points[:,:2]).reshape((1,1,2))

		# Step 1: Undistort points
		if len(points) > 0:
			points_undistorted = cv2.undistortPoints(points, self.mtx, self.dist)
		points_undistorted = np.squeeze(points_undistorted, axis=1)
		result = []
		x = (points_undistorted[0, 0] - c_x) / f_x * z
		y = (points_undistorted[0, 1] - c_y) / f_y * z
		result = ([x, y, z])
		return result
	def draw_recordframe(self,num_recorded_points):
		self.draw_circle()
		if self.calculate_distance() != None:
			point = self.undistort_point()
			point = (str(int(point[0])) +" "+ str(int(point[1])) +" "+ str(int(point[2])))
		else:
			point = ('0 0 0')
		cv2.putText(self.image,"Recorded Positions: " + str(num_recorded_points), 
		    (10,420), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0),3)
		cv2.putText(self.image,"Pos: " + point, 
		    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0),3)
		return self.image



class CameraCalibration:
	def __init__(self):
		self.img_num = 0

		# Arrays to store object points and image points from all the images.
		self.objpoints = [] # 3d point in real world space
		self.imgpoints = [] # 2d points in image plane.

		# termination criteria
		self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		self.objp = np.zeros((7*9,3), np.float32)
		self.objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)*20
		self.objp = self.objp.reshape(-1,1,3)
	def get_imagenum(self):
		return self.img_num

	def draw_imagenum(self,image):
		cv2.putText(image,"Image Number: " + str(self.img_num), 
		    (10,420), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0),3)
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
		corners = np.int0(corners)

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (9,7),flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

		# If found, add object points, image points (after refining them)
		if ret == True:
			chessboardrawn = cv2.drawChessboardCorners(image, (9,7), corners,ret)
		return image

	def appendchessboardpoints(self,image):
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
		corners = np.int0(corners)

		for i in corners:
		    x,y = i.ravel()
		    cv2.circle(image,(x,y),3,(0,0,255),-1)

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (9,7),flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

		# If found, add object points, image points (after refining them)
		if ret == True:
		    self.objpoints.append(self.objp)

		    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
		    self.imgpoints.append(corners)

		    # Draw and display the corners
		    chessboardrawn = cv2.drawChessboardCorners(image, (9,7), corners,ret)
		    self.img_num = self.img_num + 1
	def process_calibration(self,image):
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, image.shape[::-1][1:3],None,None)
		tot_error = 0
		for i in range(len(self.objpoints)):
		    imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		    error = cv2.norm(self.imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		    tot_error += error
		return mtx, dist, tot_error

class ObjectVariables():
	def __init__(self):
		self.hsv_tresholds = self.loadhsv()
		self.objectsize = self.loadobjectsize()
		self.dist,self.mtx,self.needscalibration = self.loadcalibration()
		self.capture = cv2.VideoCapture(0)
		self.recordedpositions = []
		self.viewoutput = 'original'
		self.takeimage = False
		self.recordbool = False
		self.takesnapshot = False
	def loadhsv(self):
		store = JsonStore('Saved_variables.json')
		if store.exists('variables'):
			h_min = store.get('variables')['h_min']
			s_min = store.get('variables')['s_min']
			v_min = store.get('variables')['v_min']
			h_max = store.get('variables')['h_max']
			s_max = store.get('variables')['s_max']
			v_max = store.get('variables')['v_max']
		else:
			h_min = 0
			s_min = 0
			v_min = 0
			h_max = 255
			s_max = 255
			v_max = 255
		return np.array([h_min,s_min,v_min,h_max,s_max,v_max])
	def savehsv(self):
		h_min,s_min,v_min,h_max,s_max,v_max =(
		self.hsv_tresholds[0],self.hsv_tresholds[1],self.hsv_tresholds[2],
			self.hsv_tresholds[3],self.hsv_tresholds[4],self.hsv_tresholds[5])
		store = JsonStore('Saved_variables.json')
		store.put('variables', h_min = int(h_min),s_min = int(s_min),v_min = int(v_min),
			h_max = int(h_max),s_max = int(s_max),v_max = int(v_max))
	def loadobjectsize(self):
		store = JsonStore('Saved_variables.json')
		if store.exists('ballsize'):
			ballsize = store.get('ballsize')['ballsize']
		else:
			ballsize = 6.6
		return ballsize
	def saveobjectsize(self):
		store = JsonStore('Saved_variables.json')
		store.put('ballsize',ballsize = self.objectsize)
	def loadcalibration(self):

		cameracalibration = JsonStore('Saved_calibration.json')
		if cameracalibration.exists('calibration_var'):
			mtx = cameracalibration.get('calibration_var')['mtx']
			dist = cameracalibration.get('calibration_var')['dist']
			mtx = np.array(mtx)
			dist = np.array(dist)
			needscalibration = False

		else:
			dist = None
			mtx = None
			needscalibration = True
		return dist,mtx,needscalibration
	def savecalibration(self,mtx,dist):
		calibrationstore = JsonStore('Saved_calibration.json')
		lstmtx = mtx.tolist()
		lstdist = dist.tolist()
		calibrationstore.put('calibration_var', mtx = lstmtx, dist = lstdist)
	def gethsv(self):
		return self.hsv_tresholds
	def sethsv(self,hsv):
		self.hsv_tresholds = hsv
	def getobjectsize(self):
		return self.objectsize
	def setobjectsize(self,size):
		self.objectsize = size
	def getdist(self):
		return self.dist
	def setdist(self,dist):
		self.dist = dist
	def getmtx(self):
		return self.mtx
	def setmtx(self,mtx):
		self.mtx = mtx
	def settakeimage(self, boolean):
		self.takeimage = boolean
	def gettakeimage(self):
		return self.takeimage
	def setrecordbool(self, boolean):
		self.recordbool = boolean
	def getrecordbool(self):
		return self.recordbool
	def settakesnapshot(self, boolean):
		self.takesnapshot = boolean
	def setviewoutput(self, view):
		self.viewoutput = view
	def getviewoutput(self):
		return self.viewoutput
	def getneedscalibration(self):
		return self.needscalibration
	def setneedscalibration(self,boolean):
		self.needscalibration = boolean
	def getcapture(self):
		return self.capture
	def setcapture(self,capture):
		self.capture = capture
	def capturerelease(self):
		self.capture.release()
	def appendpostionvalue(self,position):
		self.recordedpositions.append(position)
	def clearrecordedpositions(self):
		self.recordedpositions = []
	def getrecordedposition(self):
		return self.recordedpositions
	def saverecordedpositions(self,filename):
		postionsstore = JsonStore(filename +'.json')
		postionsstore.put('3dpostions_list', postionvalues = self.recordedpositions)


