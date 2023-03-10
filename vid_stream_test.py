# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
vs = VideoStream(usePiCamera=True).start()
#vs = VideoStream(src=0).start()
time.sleep(2.0)

def detect_motion():
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock
	# initialize the motion detector and the total number of frames
	# read thus far
	#md = SingleMotionDetector(accumWeight=0.1)
	total = 0
    # loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)
		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		#cv2.putText(frame, timestamp.strftime(
		#	"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            # if the total number of frames has reached a sufficient

		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()

def generate():
	# grab global references to the output frame and lock variables
    global outputFrame, lock
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
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # start a thread that will perform motion detection
    #t = threading.Thread(target=detect_motion)
    #t.daemon = True
    #t.start()

    # start the flask app
    app.run(host="0.0.0.0", port="80", debug=True, threaded=True, use_reloader=False)


# release the video stream pointer
vs.stop()