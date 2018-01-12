import serial
import time

class RotateControl:
    def rotate_to_region(self, region):
        ser = serial.Serial('/dev/ttyACM1', 9600)
        # serin = ser.read()
        if region == 1:
            ser.write('a')
        elif region == 2:
            ser.write('c')
        elif region == 3:
            ser.write('b')
        elif region == 4:
            ser.write('d')
        time.sleep(15)
        # ser.read()
        # ser.close()
