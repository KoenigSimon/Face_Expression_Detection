from picamera import PiCamera
import time

number = 0

camera = PiCamera()
camera.resolution = (640, 480)
#camera.sensor_mode = 2
time.sleep(2)
print("Cam init done, ready to take pics")

if __name__ == "__main__":

    try:
        while True:
            if input():
                camera.capture(f"./{number}.jpg")
                print(f"Captured image{number}")
                number += 1
    except KeyboardInterrupt:
        pass

    