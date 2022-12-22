from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import threading

from flask import Response
from flask import Flask
from flask import render_template

import expression_functions as expr

model_eye = './models/eye_predictor.dat'
model_mouth = './models/mouth_predictor.dat'

ipc_pipe_file_path = '../ProtoTracerRPI/data_pipe_fifo'

DEBUG_HOST_VIDEO_STREAM = False      #will only send data to ipc when false
MAX_IPC_CYCLES_PER_SEC = 30
MAX_CAMERA_FRAMERATE = 60
MAX_IPC_MESSAGE_QUEUE_SIZE = 100    #after that flush buffer

#expression detection parameters
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
LE_COUNT = 0
RE_COUNT = 0
LE_STATE = True
RE_STATE = True

MOUTH_THRESH = 3    #min open
MOUTH_MAX = 40      #max open
MOUTH_STATE = False
SMILE_THRESH = 115  #start smile
SMILE_MIN = 100     #full smile
SMILE_STATE = False

#camera calibration params
camera_matrix = np.array(
    [[535.68751981,   0.0,         325.61866707],
     [  0.0,         532.16539606, 234.24546875],
     [  0.0,           0.0,          1.0        ]])
distortion_coefficients = np.array(
    [[-0.88958658,  1.02578331,  0.01243967,  0.00313227, -0.50098573]])
camera_size = (640, 480)
scaled_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coefficients, camera_size, 1.0, camera_size)
roi_x, roi_y, roi_w, roi_h = roi

ipc_data_queue = []
outputFrame = None
app = Flask(__name__)

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictors...")
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(model_to_use)
predictor_mouth = dlib.shape_predictor(model_mouth)
predictor_eye = dlib.shape_predictor(model_eye)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=True, resolution=(640, 480), framerate=60).start()
time.sleep(2.0)

def update_video():
    # loop over the frames from the video stream
    global outputFrame
    global LE_COUNT
    global RE_COUNT
    global LE_STATE
    global RE_STATE
    global MOUTH_STATE
    global SMILE_STATE

    exec_time_start = time.time()
    last_msg_time = time.time()
    allow_msg = True

    while True:
        #measure time
        exec_time_start = time.time()

        # grab the frame from the video stream, resize it to have a
        # maximum width of 400 pixels, and convert it to grayscale
        frame = vs.read()
        frame = cv2.flip(frame, 0)

        #distortion correction of closeup fisheye camera        
        undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None, scaled_camera_matrix)        
        cropped_frame = undistorted_frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]    
        frame = imutils.resize(cropped_frame, width=300) 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #color = np.zeros((400, 400, 3), np.uint8)

        # fixed rect because fixed camera position, so using face detection not necessary
        rect = dlib.rectangle(left=70, top=20, right=230, bottom=250)

        # convert the dlib rectangle into an OpenCV bounding box and
        # draw a bounding box surrounding the face
        if DEBUG_HOST_VIDEO_STREAM:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # use our custom dlib shape predictor to predict the location
        # of our landmark coordinates, then convert the prediction to
        # an easily parsable NumPy array
        shape_mouth = predictor_mouth(gray, rect)
        shape_mouth = face_utils.shape_to_np(shape_mouth)
        shape_eye = predictor_eye(gray, rect)
        shape_eye = face_utils.shape_to_np(shape_eye)

        # loop over the (x, y)-coordinates from our dlib shape
        # predictor model draw them on the image
        if DEBUG_HOST_VIDEO_STREAM:
            count = 0
            for (sX, sY) in shape_mouth:
                cv2.circle(frame, (sX, sY), 1, (0, 255, (count*10)), -1)
                count += 1
            for (sX, sY) in shape_eye:
                cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
            outputFrame = frame

        #do expression extraction
        ear_right = expr.eye_aspect_ratio(shape_eye[0:6])
        ear_left = expr.eye_aspect_ratio(shape_eye[6:12])

        #smile between 120+neutral and 110-smile
        #opennes between 5-closed and 40-open
        opennes, smile_fac = expr.mouth_expressions(shape_mouth)

        #do expression processing / state machine
        #limit messages per sec
        if last_msg_time + (1.0/MAX_IPC_CYCLES_PER_SEC) < time.time():
            last_msg_time = time.time()
            allow_msg = True

        if ear_left < EYE_AR_THRESH: LE_COUNT += 1
        else: LE_COUNT -= 1
        if LE_COUNT > EYE_AR_CONSEC_FRAMES:
            if allow_msg and LE_STATE == True:
                LE_STATE = False
                write_pipe("le:0")
            LE_COUNT = EYE_AR_CONSEC_FRAMES
        elif LE_COUNT < 0:
            if allow_msg and LE_STATE == False:
                LE_STATE = True
                write_pipe("le:1")
            LE_COUNT = 0

        if ear_right < EYE_AR_THRESH: RE_COUNT += 1
        else: RE_COUNT -= 1
        if RE_COUNT > EYE_AR_CONSEC_FRAMES:
            if allow_msg and  RE_STATE == True: 
                RE_STATE = False
                write_pipe("re:0")
            RE_COUNT = EYE_AR_CONSEC_FRAMES
        elif RE_COUNT < 0:
            if allow_msg and RE_STATE == False:
                RE_STATE = True
                write_pipe("re:1")
            RE_COUNT = 0

        if allow_msg and opennes > MOUTH_THRESH:
            write_pipe(f"mo:{ round(opennes/MOUTH_MAX, 2)}")
            MOUTH_STATE = True
        elif allow_msg and MOUTH_STATE == True:
            write_pipe("mo:0.0")
            MOUTH_STATE = False

        if allow_msg and smile_fac < SMILE_THRESH:
            write_pipe(f"ms:{ round((smile_fac - SMILE_MIN) / (SMILE_THRESH - SMILE_MIN), 2) }")
            SMILE_STATE = True
        elif allow_msg and SMILE_STATE == True:
            write_pipe("ms:0.0")
            SMILE_STATE = False

        allow_msg = False


        #limit fps
        exec_time_end = time.time()
        exec_time = exec_time_end - exec_time_start
        delta_t = (1.0/60.0) - exec_time
        if delta_t > 0: 
            time.sleep(delta_t)
            print(f"FPS: { round(1.0 / (exec_time + delta_t)) }", end="\r")
        else:
            print(f"FPS: { round(1.0 / (exec_time)) }", end="\r")

def write_pipe(data):
    if DEBUG_HOST_VIDEO_STREAM:
        print(data)
    ipc_data_queue.append(data)
    if len(ipc_data_queue) > MAX_IPC_MESSAGE_QUEUE_SIZE: #some app not running to read the buffer, clear it
        ipc_data_queue.clear()
        print("Clearing Message Queue cuz full")

def grab_frame():
    # grab global references to the output frame and lock variables
    global outputFrame
    # loop over frames from the output stream
    while True:
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(grab_frame(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":

    #start video thread
    t = threading.Thread(target=update_video)
    t.daemon = True
    t.start()

    # start the flask app
    if DEBUG_HOST_VIDEO_STREAM:
        app.run(host="0.0.0.0", port="80", debug=True, threaded=True, use_reloader=False)
    else:
        try:
            while True:
                #write processed data to pipe for prototracer to read
                with open(ipc_pipe_file_path, 'w') as pipe:
                    if len(ipc_data_queue) > 0:
                        data = ipc_data_queue.pop(0)
                        pipe.write(data)
                        pipe.flush()
                pipe.close()
        except KeyboardInterrupt:
            pass

# do a bit of cleanup
#cv2.destroyAllWindows()
vs.stop()
print("")