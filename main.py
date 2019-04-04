import numpy as np
import imutils
import argparse
from collections import deque
from kivy.app import App
from kivy.storage.jsonstore import JsonStore
from kivy.base import EventLoop
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.lang import Builder
import cv2

Builder.load_string('''
<SourceScreen>
	on_enter: root.dostart()
	#on_pre_leave: root.doexit()
	BoxLayout:
		orientation: 'vertical'
		KivyCamera:
			id: sourcecam
		BoxLayout:
			id: sourcebuttonbox
			size_hint_y : None
			Button:
	            text: "camera calibration"
	            on_press: root.manager.current = 'Camera Calibration Screen'
	        Button:
	            text: "Source"

	        Button:
	            text: "mask calibration"
	            on_press:root.manager.current = 'Calibration Screen'
	        Button:
	            text: "Result"
	            on_press: root.manager.current = 'Result Screen'
	        Button:
	            text: "Record"
	            on_press:root.manager.current = 'Record Screen'

<CalibrationScreen>:
	on_enter: root.dostart()
	on_pre_leave: root.doexit()
	BoxLayout:
		orientation: 'vertical'
		KivyCamera:
			id: maskcam
		BoxLayout:
			size_hint_y : None
	        Button:
	            text: "Source"
	            on_press: root.manager.current = 'Source Screen'
	        Button:
	            text: "change ball size"
	            on_press: root.changeballsizebtn()
	        Button:
	            text: "Result"
	            on_press: root.manager.current = 'Result Screen'
	    BoxLayout:
	    	size_hint_y : None
	    	orientation: 'vertical'
	    	Label:
	    		id:Hue_values
	    		text: 'Hue values (Max/Min)'
			Slider:
				id:H_max_slider
				value: root.hmax_val
				min: 0
				max: 255
				on_value:root.new_hmax_val(*args)
			Slider:
				id:H_min_slider
				value: root.hmin_val
				min: 0
				max: 255
				on_value:root.new_hmin_val(*args)
	    	Label:
	    		id:Sat_dropdownbtn
	    		text: 'Saturation values (Max/Min)'
			Slider:
				id:S_max_slider
				value: root.smax_val
				min: 0
				max: 255
				on_value:root.new_smax_val(*args)
			Slider:
				id:S_min_slider
				value: root.smin_val
				min: 0
				max: 255
				on_value:root.new_smin_val(*args)
	    	Label:
	    		id:Hue_dropdownbtn
	    		text: 'Value values (Max/Min)'
			Slider:
				id:V_max_slider
				value: root.vmax_val
				min: 0
				max: 255
				on_value:root.new_vmax_val(*args)
			Slider:
				id:V_min_slider
				value: root.vmin_val
				min: 0
				max: 255
				on_value:root.new_vmin_val(*args)


<ResultScreen>
	on_enter: root.dostart()
	#on_pre_leave: root.doexit()
	BoxLayout:
		orientation: 'vertical'
		KivyCamera:
			id: resultcam
		BoxLayout:
			size_hint_y : None
			Button:
	            text: "camera calibration"
	            on_press: root.manager.current = 'Camera Calibration Screen'
	        Button:
	            text: "Source"
	            on_press: root.manager.current = 'Source Screen'
	        Button:
	            text: "mask calibration"
	            on_press:root.manager.current = 'Calibration Screen'
	        Button:
	            text: "Result"
	        Button:
	            text: "Record"
	            on_press:root.manager.current = 'Record Screen'


<RecordScreen>
	on_enter: root.dostart()
	#on_pre_leave: root.doexit()
	BoxLayout:
		orientation: 'vertical'
		KivyCamera:
			id: recordcam
		BoxLayout:
			size_hint_y : None
			Button:
	            text: "Start recording every frame"
	            on_press: root.startrecord()
	        Button:
	            text: "Record single frame"
	            on_press: root.takesnapshot()
	        Button:
	            text: "Save recorded values"
	            on_press: root.endrecord()
	        Button:
	            text: "Back to Source"
	            on_press: root.manager.current = 'Source Screen'
<CameraCalibrationScreen>
	on_enter: root.dostart()
	#on_pre_leave: root.doexit()
	BoxLayout:
		orientation: 'vertical'
		KivyCamera:
			id: calibratecam
		BoxLayout:
			size_hint_y : None
			Button:
	            text: "Begin Camera calibration"
	            on_press: root.manager.current = 'Calibration in progress Screen'
	        Button:
	            text: "Source"
	            on_press: root.manager.current = 'Source Screen'
<CalibrationinprogressScreen>
	on_enter: root.dostart()
	#on_pre_leave: root.doexit()
	BoxLayout:
		orientation: 'vertical'
		KivyCamera:
			id: calibrateinprogresscam
		BoxLayout:
			size_hint_y : None
			Button:
		        text: "Take photo"
		        on_press: root.takephoto()
		    Button:
		        text: "Process calibration"
		        on_press: root.Processcalibration()
		    Button:
		        text: "Clear calibration"
		        on_press: root.clearcalibration()
		BoxLayout:
			size_hint_y : None
			Button:
		        text: "Back to source screen"
		        on_press: root.manager.current = 'Source Screen'
<ballsizeinput>
	title: 'ballsizeinput'
	size_hint: None, None
	size: 400, 120
	text: input.text
	BoxLayout:
		orientation: 'vertical'
        pos: self.pos
        size: root.size

        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'Enter Ball size (cm)'
            TextInput:
                id: input
                multiline: False
                hint_text:'Ball size (cm)'
        BoxLayout:
            orientation: 'horizontal'
            Button:
                text: 'Enter'
                on_press: root._enter()

<filenameinput>
	title: 'filenameinput'
	size_hint: None, None
	size: 400, 120
	text: input.text
	BoxLayout:
		orientation: 'vertical'
        pos: self.pos
        size: root.size

        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'Enter Filename'

            TextInput:
                id: input
                multiline: False
                hint_text:'Filename'
        BoxLayout:
            orientation: 'horizontal'
            Button:
                text: 'Enter'
                on_press: root._enter()
''')

global h_min, s_min, v_min, h_max, s_max, v_max
store = JsonStore('Saved_variables.json')
if store.exists('variables'):
	h_min = store.get('variables')['h_min']
	s_min = store.get('variables')['s_min']
	v_min = store.get('variables')['v_min']
	h_max = store.get('variables')['h_max']
	s_max = store.get('variables')['s_max']
	v_max = store.get('variables')['v_max']
	ballsize = store.get('ballsize')['ballsize']
else:
	h_min = 0
	s_min = 0
	v_min = 0
	h_max = 255
	s_max = 255
	v_max = 255
	ballsize = 6.6
cameracalibration = JsonStore('Saved_calibration.json')
global dist,mtx,dist,needscalibration
if cameracalibration.exists('calibration_var'):
    mtx = cameracalibration.get('calibration_var')['mtx']
    dist = cameracalibration.get('calibration_var')['dist']
    mtx = np.array(mtx)
    dist = np.array(dist)
    needscalibration = False

else:
	needscalibration = True
	

capture = cv2.VideoCapture(0)
viewoutput = 'original'
allpositionvalues = []
	    
class KivyCamera(Image):
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None
        self.clock_variable = None
        
        #init various global vals
        global takeimage, img_num, recordbool,takesnapshot
        recordbool,takesnapshot,takeimage,img_num = False,False,False,0

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((7*9,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)*20
        self.objp = self.objp.reshape(-1,1,3)

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    	# Arrays to store object points and image points from all the images.
        global objpoints, imgpoints
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.


    def start(self, capture, fps=30):
        self.capture = capture
        if self.clock_variable is None:
        	self.clock_variable = Clock.schedule_interval(self.update, 1.0 / fps)
        else:
	        self.clock_variable.cancel()
	        self.clock_variable = Clock.schedule_interval(self.update, 1.0 / fps)
        

    def stop(self):
        Clock.unschedule(self.update)
        self.capture = None
        self.clock_variable = None


    def update(self, dt):
    	def displayimage(image):
            texture = self.texture #get kivy texture from class
            try:
                w, h = image.shape[1], image.shape[0]  #get image parameters
            except AttributeError: # attribute error catch for when a frame is empty and so not to crash program
                w, h=None,None
            if not texture or texture.width != w or texture.height != h: #resize if texture is empty or not matching image size 
	            self.texture = texture = Texture.create(size=(w, h))
	            texture.flip_vertical()
            texture.blit_buffer(image.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()
    	def recordpositions(x,y,z,frame):
            global firstframe, allpositionvalues
            if firstframe == True:
                global initx, inity, initz
                allpositionvalues = []
                distorted_point = np.array([[x,y]], dtype=np.float32)
                initx, inity, initz, undistorted_point = Unproject(distorted_point,z,mtx,dist,0,0,0,firstframe)
                allpositionvalues.append(undistorted_point)
                firstframe = False
                print("first frame")
            else:
                distorted_point = np.array([[x,y]], dtype=np.float32)
                undistorted_point = Unproject(distorted_point,z,mtx,dist,initx, inity, initz)          
                allpositionvalues.append(undistorted_point)
                #display XYZ coords on screen for debug
                # cv2.putText(frame, "posx:%.2f" % (undistorted_point[0][0]),
                # (frame.shape[1] - 300, frame.shape[0] - 200), cv2.FONT_HERSHEY_SIMPLEX,
                # 2.0, (0, 255, 0), 3)
                # cv2.putText(frame, "posY:%.2f" % (undistorted_point[0][1] / 12),
                # 	(frame.shape[1] - 300, frame.shape[0] - 170), cv2.FONT_HERSHEY_SIMPLEX,
                # 	2.0, (0, 255, 0), 3)
                # cv2.putText(frame, "posZ:%.2f" % (undistorted_point[0][2] / 12),
                # 	(frame.shape[1] - 300, frame.shape[0] - 130), cv2.FONT_HERSHEY_SIMPLEX,
                # 	2.0, (0, 255, 0), 3)
                print("second frame")
            displayimage(frame)


    	def calculate_distance(width_of_object, focalLength, detectedwidth):
	    	
	    	return ((float(width_of_object) * float(focalLength)) / float(detectedwidth))
    	def Unproject(points, z, matrix, distortion, init_x, init_y, init_z,firstframe = False):
	    	f_x = matrix[0, 0]
	    	f_y = matrix[1, 1]
	    	c_x = matrix[0, 2]
	    	c_y = matrix[1, 2]

	    	# Step 1: Undistort points
	    	points_undistorted = np.array([])
	    	num_pts = points.size / 2
	    	points.shape = (int(num_pts), 1, 2)
	    	if len(points) > 0:
	    		points_undistorted = cv2.undistortPoints(points, matrix, distortion)
	    	points_undistorted = np.squeeze(points_undistorted, axis=1)
	    	result = []
	    	if firstframe == True:
	    		for i in range(points_undistorted.shape[0]):
		    		x = (points_undistorted[i, 0] - c_x) / f_x * z
		    		y = (points_undistorted[i, 1] - c_y) / f_y * z
		    		result.append([x-x, y-y, z-z])
		    	return x, y, z, result
	    	elif firstframe == False:
		    	for i in range(points_undistorted.shape[0]):
		    		x = (points_undistorted[i, 0] - c_x) / f_x * z
		    		y = (points_undistorted[i, 1] - c_y) / f_y * z
		    		#x = points_undistorted[i, 0]
		    		#y = points_undistorted[i, 1]
		    		result.append([x-initx, y-inity, z-initz])
		    		print(x-initx, y-inity, z-initz)
		    	return result
    	global calibratecameraoption
    	if calibratecameraoption ==  True:
		    _, frame = self.capture.read()
		    global imgshape
		    imgshape = frame
		    textimg = frame

		    global takeimage, img_num

		    cv2.putText(textimg,"Image Number: " + str(img_num), 
		    (10,420), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0),3)

		    
		    if takeimage == True:
		        takeimage = False
		        img = frame
		        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		        corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
		        corners = np.int0(corners)

		        for i in corners:
		            x,y = i.ravel()
		            cv2.circle(frame,(x,y),3,(0,0,255),-1)

		        # Find the chess board corners
		        ret, corners = cv2.findChessboardCorners(gray, (9,7),flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

		        # If found, add object points, image points (after refining them)
		        if ret == True:
		            global objpoints, imgpoints
		            objpoints.append(self.objp)

		            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
		            imgpoints.append(corners)

		            # Draw and display the corners
		            img = cv2.drawChessboardCorners(img, (9,7), corners,ret)
		            img_num = img_num + 1
		            displayimage(img)
		            
		        
		        else:
		            displayimage(textimg)
		    else:
		    	displayimage(textimg)
    	else:
	    	global viewoutput
	    	ret, frame = self.capture.read()
	    	_, sourcecam = self.capture.read()
	    	if ret:
	    		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		    	lower_threshold = np.array([h_min,s_min,v_min])
		    	upper_threshold = np.array([h_max,s_max,v_max])

		    	mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
		    	#mask = cv2.GaussianBlur(mask, (11, 11), 0)

		    	kernel = np.ones((5,5),np.uint8)
		    	res = cv2.bitwise_and(frame,frame, mask= mask)

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
		    			



		    		diameter = int(radius) * 2

			    	# only proceed if the radius meets a minimum size
			    	if radius > 10:# draw the circle and centroid on the result,
			    		cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			    		cv2.circle(frame, center, 5, (0, 0, 255), -1)
		    	if viewoutput == 'original':# convert it to texture
		            displayimage(sourcecam)
		    	elif viewoutput == 'mask':
		            displayimage(res)
		    	elif viewoutput == 'result':
		            global mtx, ballsize
		            F = mtx[0, 0]
		            W = ballsize/10
		            try: 
		            	distance = calculate_distance(F,W,diameter)
		            except UnboundLocalError:
		            	distance = 0
		            except ZeroDivisionError:
		            	distance = 0
		            cv2.putText(frame, "%.2fM" % (distance / 12),
		            (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		            2.0, (0, 255, 0), 3)
		            displayimage(frame)
		    	elif viewoutput == 'record':
		    		global recordbool,takesnapshot
		    		print("record screen")
		    		F = mtx[0, 0]
		    		W = ballsize/10
		    		try: 
		    			distance = calculate_distance(F,W,diameter)
		    		except UnboundLocalError:
		    			distance = 0
		    		except ZeroDivisionError:
		    			distance = 0
		    		if recordbool == True:
		    			print("true")
		    			try:
		    				recordpositions(x,y,distance,frame)
		    			except UnboundLocalError:
		    				print("unbound")
		    				pass
		    		elif takesnapshot == True:
		    			takesnapshot = False
		    			try:
		    				recordpositions(x,y,distance,frame)
		    			except UnboundLocalError:
		    				print("unbound")
		    		else:
		    			print("false")
		    			displayimage(frame)





class SourceScreen(Screen):
	def dostart(self, *largs):
		if needscalibration == True:
			popup = Popup(title='Camera Calibration error',
			    content=Label(text='''Your camera is not calibrated,\nplease open "camera calibration" and\nuse chessboard pattern to calibrate camera'''),
			    size_hint=(None, None), size=(400, 400))
			popup.open()
		global calibratecameraoption
		calibratecameraoption = False
		global capture
		global viewoutput
		viewoutput = 'original'
		self.ids.sourcecam.start(capture)
	def doexit(self):
		self.ids.sourcecam.stop()

		

class CalibrationScreen(Screen):
	global h_max,h_min,s_min,s_max,v_max,v_min
	hmax_val,hmin_val,smax_val,smin_val,vmax_val,vmin_val = h_max,h_min,s_max,s_min,v_max,v_min
	def dostart(self, *largs):
		global capture
		global viewoutput
		viewoutput = 'mask'
		self.ids.maskcam.start(capture)
	def doexit(self):
		self.ids.maskcam.stop()
		global store,hmax_val,hmin_val,smax_val,smin_val,vmax_val,vmin_val
		store.put('variables', h_min = h_min,s_min = s_min,v_min = v_min,
			h_max = h_max,s_max = s_max,v_max = v_max)
	def changeballsizebtn(self):
		obj = ballsizeinput(self)
		obj.open()

	def new_hmax_val(self, *args):
		global h_max
		h_max = int(args[1])
		self.hmax_val = h_max
	def new_hmin_val(self, *args):
		global h_min
		h_min = int(args[1])
		self.hmin_val = h_min       
	def new_smin_val(self, *args):
		global s_min
		s_min = int(args[1])
		self.smin_val = s_min
	def new_smax_val(self, *args):
		global s_max
		s_max = int(args[1])
		self.smax_val = s_max
	def new_vmax_val(self, *args):
		global v_max
		v_max = int(args[1])
		self.vmax_val = v_max
	def new_vmin_val(self, *args):
		global v_min
		v_min = int(args[1])
		self.vmin_val = v_min
class ResultScreen(Screen):
	def dostart(self, *largs):
		global capture
		global viewoutput
		viewoutput = 'result'
		capture = cv2.VideoCapture(0)
		self.ids.resultcam.start(capture)
	def doexit(self):
		self.ids.resultcam.stop()
class RecordScreen(Screen):
		def dostart(self, *largs):
			global firstframe
			firstframe = True
			global capture
			global viewoutput
			viewoutput = 'record'
			self.ids.recordcam.start(capture)
		def doexit(self):
			self.ids.recordcam.stop()
		def startrecord(self):
			global recordbool
			recordbool = True
		def takesnapshot(self):
			global takesnapshot
			takesnapshot = True
		def endrecord(self):
			global recordbool, outputpositionsfilename,firstframe
			firstframe = True
			recordbool = False
			obj = filenameinput(self)
			obj.open()
class ballsizeinput(Popup):			
	def __init__(self, parent, *args):
		super(ballsizeinput, self).__init__(*args)
	def _enter(self):
		global ballsize
		if not self.text:
			popup = Popup(title='input error',content=Label(text='please input ball size'),
            	size_hint=(None, None), size=(400, 400))
			popup.open()
		else:
			ballsize = float(self.text)
			self.dismiss()
			store = JsonStore('Saved_variables.json')
			ballsize = store.put('ballsize',ballsize = ballsize)
class filenameinput(Popup):
	def __init__(self, parent, *args):
		super(filenameinput, self).__init__(*args)


	def _enter(self):
		global outputpositionsfilename, allpositionvalues
		if not self.text:
			popup = Popup(title='input error',content=Label(text='please input filename'),
            	size_hint=(None, None), size=(400, 400))
			popup.open()
		else:
			outputpositionsfilename = str(self.text)
			self.dismiss()
			postionsstore = JsonStore(outputpositionsfilename +'.json')
			postionsstore.put('3dpostions_list', postionvalues = allpositionvalues)
			popup2 = Popup(title='Save confirmation',content=Label(text='3D positional values saved to:\n'+str(outputpositionsfilename)+'.json'),
            	size_hint=(None, None), size=(400, 400))
			popup2.open()



class CameraCalibrationScreen(Screen):
	def dostart(self, *largs):
		global capture
		global calibratecameraoption
		calibratecameraoption = True
		self.ids.calibratecam.start(capture)
		pass
	def doexit(self):
		pass
class CalibrationinprogressScreen(Screen):
	def dostart(self, *largs):
		global capture
		global viewoutput
		capture = cv2.VideoCapture(0)
		self.ids.calibrateinprogresscam.start(capture)
		popup3 = Popup(title='Camera Calibration explaination',content=Label(text='Your camera is now ready to be calibrated,\nprint out camera calibration chessboard and\n take at least 20 images of the chessboard\n at a variety of angles and distances to the camera'),
		 size_hint=(None, None), size=(400, 400))
		popup3.open()
	def doexit(self):
		pass
	def takephoto(self):
		global takeimage
		takeimage = True
	def Processcalibration(self):
		global objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs, imgshape, needscalibration
		needscalibration = False
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgshape.shape[::-1][1:3],None,None)
		tot_error = 0
		for i in range(len(objpoints)):
		    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		    tot_error += error
		calibrationstore = JsonStore('Saved_calibration.json')
		lstmtx = mtx.tolist()
		lstdist = dist.tolist()

		calibrationstore.put('calibration_var', mtx = lstmtx, dist = lstdist)
		popup2 = Popup(title='Camera Calibration total error',
			    content=Label(text='Your camera is now Calibrated,\nThe total calculated error is:\n'+str(tot_error)+'\nit is suggested that the error be no more than 2\nif your error is too high\nplease calibrate again'),
			    size_hint=(None, None), size=(400, 400))
		popup2.open()

	def clearcalibration(self):
		global objpoints, imgpoints, img_num
		img_num = 0
		objpoints = []
		imgpoints = []

		

sm = ScreenManager()
#init screens
sm.add_widget(SourceScreen(name="Source Screen"))
sm.add_widget(CalibrationScreen(name="Calibration Screen"))
sm.add_widget(ResultScreen(name="Result Screen"))
sm.add_widget(RecordScreen(name="Record Screen"))
sm.add_widget(CameraCalibrationScreen(name="Camera Calibration Screen"))
sm.add_widget(CalibrationinprogressScreen(name="Calibration in progress Screen"))


class CamApp(App):		

	def __init__(self):
		super(CamApp,self).__init__()
		KivyCamera.viewoutput = 'original'
		pass

	def build(self):
		return sm
	def on_stop(self):
		capture.release()

if __name__ == '__main__':
    CamApp().run()