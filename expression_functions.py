from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def mouth_expressions(mouth):
    #opennes measured from inner lips
    opennes = (dist.euclidean(mouth[13], mouth[19]) + dist.euclidean(mouth[14], mouth[18]) + dist.euclidean(mouth[15], mouth[17])) / 3.0
    
    #smile
    #compare mouth corners with upper lip, when close together = smiling
    #for now only y distance, more advanced can be done later
    avg_corner_height = (mouth[0][1] + mouth[12][1] + mouth[6][1] + mouth[16][1]) / 4
    avg_upper_lip = sum(mouth[1:5][1]) / 5
    y_dist = abs(avg_corner_height - avg_upper_lip)
    
    return (opennes, y_dist)