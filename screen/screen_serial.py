#!/usr/bin/python
# -*- coding: <utf-8> -*-

#/*
#    @year:        2020/2021
#    @author:      Sekomer
#    @touch:       aksoz19@itu.edu.tr
#*/

import serial
import rospy
from sensor_msgs.msg import LaserScan
from rosserial_arduino.msg import Adc
import os
import sys
import time 

arduino = serial.Serial('/dev/ttyACM1', baudrate=57600, timeout=0.1)  
time.sleep(0.5)


    
def mapper(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
            

def callback(data):
	speed = str(mapper(data.adc0, 0, 1000, 0, 99)).zfill(2)
	steer = str(data.adc1).zfill(4)
	regen = str(mapper(data.adc2, 0, 1000, 0, 99)).zfill(2)
	gear = str(data.adc3)		
	auto  = str(data.adc4)
	#extra = "0"
	
	yolla = speed + steer + regen + gear + auto + "*"
	arduino.write(bytes(yolla, "utf-8"))
	rospy.sleep(0.1)
    
    
if __name__ == "__main__":
	rospy.init_node("screen_serial_node", anonymous=True)
	rospy.Subscriber("screen", Adc, callback)
