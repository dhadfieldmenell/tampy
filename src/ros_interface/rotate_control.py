import serial
import time

class RotateControl:
    def rotate_to_region(self, region, timeout=20):
        ser = serial.Serial('/dev/ttyACM1', 9600)
        # # serin = ser.read()
        # if region == 1:
        #     ser.write('a')
        # elif region == 2:
        #     ser.write('c')
        # elif region == 3:
        #     ser.write('b')
        # elif region == 4:
        #     ser.write('d')
        # if region == 1:
        #     ser.write('d')
        # elif region == 2:
        #     ser.write('b')
        # elif region == 3:
        #     ser.write('c')
        # elif region == 4:
        #     ser.write('a')
        if region == 1:
            ser.write('d')
        elif region == 2:
            ser.write('b')
        elif region == 3:
            ser.write('a')
        elif region == 4:
            ser.write('a')
        time.sleep(timeout)
        # ser.read()
        # ser.close()
