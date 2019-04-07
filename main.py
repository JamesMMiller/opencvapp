import numpy as np
import imutils
from kivy.app import App
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
from opencvprocessing import ObjectTracker
from opencvprocessing import CameraCalibration
from opencvprocessing import ObjectVariables


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

allvariables = ObjectVariables()
	    
class KivyCamera(Image):
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None
        self.clock_variable = None
        
        #init various  vals
        self.recordbool,self.takesnapshot,self.takeimage= False,False,False

    def start(self, capture, fps=30):
        self.capture = capture
        if self.clock_variable is None:
        	self.clock_variable = Clock.schedule_interval(self.update, 1.0 / fps)
        else:
	        self.clock_variable.cancel()
	        self.clock_variable = Clock.schedule_interval(self.update, 1.0 / fps)
        

    def stop(self):
        self.capture = None
        self.clock_variable.cancel()
        self.clock_variable = None


    def update(self, dt):

    	def displayimage(image,wait=0):
    		texture = self.texture #get kivy texture from class
    		try:
    			w, h = image.shape[1], image.shape[0]  #get image parameters
    		except AttributeError: # attribute error catch for when a frame is empty and so not to crash program
    			w, h=None,None
    		if w != None: #resize if texture is empty or not matching image size 
    			if not texture or texture.width != w or texture.height != h:    
    				self.texture = texture = Texture.create(size=(w, h))
    				texture.flip_vertical()
    			texture.blit_buffer(image.tobytes(), colorfmt='bgr')
    		self.canvas.ask_update()		        

    	ret, frame = self.capture.read()
    	if allvariables.getviewoutput() == 'camera calibration':
		    if allvariables.gettakeimage() == True:
		        CalibrationinprogressScreen.calibrationobj.appendchessboardpoints(frame)
		        allvariables.settakeimage(False)
		    
		    displayimage(CalibrationinprogressScreen.calibrationobj.draw_imagenum(frame))       
    	else:
	    	needscalibration = allvariables.getneedscalibration()
	    	if ret:
	    		hsv_tresholds = allvariables.gethsv()
	    		ballsize = allvariables.getobjectsize()
	    		mtx = allvariables.getmtx()
	    		dist = allvariables.getdist()
	    		trackedobject = ObjectTracker(frame, hsv_tresholds, ballsize,mtx,dist)
		    	if allvariables.getviewoutput() == 'original':# convert it to texture
		            displayimage(frame)
		    	elif allvariables.getviewoutput() == 'mask':
		            displayimage(trackedobject.get_mask_res())
		    	elif allvariables.getviewoutput() == 'result':
		            displayimage(trackedobject.draw_circle())
		    	elif allvariables.getviewoutput() == 'record':
		    		if (allvariables.getrecordbool() == True or allvariables.gettakesnapshot() == True) and trackedobject.calculate_distance() != None:
		    			undisrortedposition = trackedobject.undistort_point()
		    			allvariables.appendpostionvalue(undisrortedposition)
		    			allvariables.settakesnapshot(Fasle)
		    			displayimage(trackedobject.draw_recordframe(len(allvariables.getrecordedposition())))
		    		else:
		    			displayimage(trackedobject.draw_recordframe(len(allvariables.getrecordedposition())))
	    	else:
		    	displayimage(CalibrationinprogressScreen.calibrationobj.draw_imagenum(frame))





class SourceScreen(Screen):
	def dostart(self, *largs):
		needscalibration = allvariables.getneedscalibration()
		if needscalibration == True:
			popup = Popup(title='Camera Calibration error',
			    content=Label(text='''Your camera is not calibrated,\nplease open "camera calibration" and\nuse chessboard pattern to calibrate camera'''),
			    size_hint=(None, None), size=(400, 400))
			popup.open()
		capture = allvariables.getcapture()
		allvariables.setviewoutput('original')
		self.ids.sourcecam.start(capture)
	def doexit(self):
		self.ids.sourcecam.stop()

		

class CalibrationScreen(Screen):
	hsv_t = allvariables.gethsv()
	hmax_val = int(hsv_t[3]) 
	hmin_val = int(hsv_t[0])
	smin_val = int(hsv_t[1]) 
	smax_val = int(hsv_t[4]) 
	vmax_val = int(hsv_t[5])
	vmin_val = int(hsv_t[2]) 


	def dostart(self, *largs):
		capture = allvariables.getcapture()
		allvariables.setviewoutput('mask')
		self.ids.maskcam.start(capture)
	def doexit(self):
		self.ids.maskcam.stop()
		allvariables.savehsv()
	def changeballsizebtn(self):
		obj = ballsizeinput(self)
		obj.open()

	def new_hmax_val(self, *args):
		self.hsv_t[3] = int(args[1])
		allvariables.sethsv(self.hsv_t)
	def new_hmin_val(self, *args):
		self.hsv_t[0] = int(args[1])
		allvariables.sethsv(self.hsv_t)
	def new_smin_val(self, *args):
		self.hsv_t[1] = int(args[1])
		allvariables.sethsv(self.hsv_t)
	def new_smax_val(self, *args):
		self.hsv_t[4] = int(args[1])
		allvariables.sethsv(self.hsv_t)
	def new_vmax_val(self, *args):
		self.hsv_t[5] = int(args[1])
		allvariables.sethsv(self.hsv_t)
	def new_vmin_val(self, *args):
		self.hsv_t[2] = int(args[1])
		allvariables.sethsv(self.hsv_t)
class ResultScreen(Screen):
	def dostart(self, *largs):
		allvariables.setviewoutput('result')
		capture = allvariables.getcapture()
		self.ids.resultcam.start(capture)
	def doexit(self):
		self.ids.resultcam.stop()
class RecordScreen(Screen):
		def dostart(self, *largs):
			self.allpositionvalues = []
			firstframe = True
			allvariables.setviewoutput('record')
			capture = allvariables.getcapture()
			self.ids.recordcam.start(capture)
		def doexit(self):
			self.ids.recordcam.stop()
		def startrecord(self):
			allvariables.setrecordbool(True)
			recordbool = True
		def takesnapshot(self):
			allvariables.settakesnapshot(True)
		def endrecord(self):
			allvariables.setrecordbool(False)
			obj = filenameinput(self)
			obj.open()
class ballsizeinput(Popup):			
	def __init__(self, parent, *args):
		super(ballsizeinput, self).__init__(*args)
	def _enter(self):
		if not self.text:
			popup = Popup(title='input error',content=Label(text='please input ball size'),
            	size_hint=(None, None), size=(400, 400))
			popup.open()
		else:
			
			allvariables.setobjectsize(float(self.text))
			allvariables.saveobjectsize()
			self.dismiss()
class filenameinput(Popup):
	def __init__(self, parent, *args):
		super(filenameinput, self).__init__(*args)

	def _enter(self):
		if not self.text:
			popup = Popup(title='input error',content=Label(text='please input filename'),
            	size_hint=(None, None), size=(400, 400))
			popup.open()
		else:
			outputpositionsfilename = str(self.text)
			self.dismiss()
			allvariables.saverecordedpositions(outputpositionsfilename)
			popup2 = Popup(title='Save confirmation',content=Label(text='3D positional values saved to:\n'+str(outputpositionsfilename)+'.json'),
            	size_hint=(None, None), size=(400, 400))
			popup2.open()



class CameraCalibrationScreen(Screen):
	def dostart(self, *largs):
		allvariables.setviewoutput('camera calibration')
		capture = allvariables.getcapture()
		self.ids.calibratecam.start(capture)
	def doexit(self):
		pass
class CalibrationinprogressScreen(Screen):
	calibrationobj = CameraCalibration()
	def dostart(self, *largs):
		mtx = allvariables.getmtx()
		dist = allvariables.getdist()
		capture = allvariables.getcapture()
		self.ids.calibrateinprogresscam.start(capture)
		popup3 = Popup(title='Camera Calibration explaination',content=Label(text='Your camera is now ready to be calibrated,\nprint out camera calibration chessboard and\n take at least 20 images of the chessboard\n at a variety of angles and distances to the camera'),
		 size_hint=(None, None), size=(400, 400))
		popup3.open()
	def doexit(self):
		pass
	def takephoto(self):
		allvariables.settakeimage(True)
	def Processcalibration(self):
		if self.calibrationobj.get_imagenum() > 10:
			needscalibration = allvariables.setneedscalibration(False)
			_,image = allvariables.getcapture().read()
			mtx, dist, tot_error = self.calibrationobj.process_calibration(image)
			allvariables.setmtx(mtx)
			allvariables.setdist(dist)
			allvariables.savecalibration(mtx,dist)

			popup2 = Popup(title='Camera Calibration total error',
				    content=Label(text='Your camera is now Calibrated,\nThe total calculated error is:\n'+str(tot_error)+'\nit is suggested that the error be no more than 2\nif your error is too high\nplease calibrate again'),
				    size_hint=(None, None), size=(400, 400))
			popup2.open()
		else:
			popup2 = Popup(title='Camera Calibration error',
				    content=Label(text='Please take at least 10 photo\nof the chessboard pattern'),
				    size_hint=(None, None), size=(400, 400))
			popup2.open()

	def clearcalibration(self):
		self.calibrationobj.__init__()

		

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
		pass

	def build(self):
		return sm
	def on_stop(self):
		allvariables.capturerelease()

if __name__ == '__main__':
    CamApp().run()