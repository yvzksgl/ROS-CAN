#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import sys
import math
from numpy.lib.function_base import average
import rospy
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import Vector3, Point
import numpy as np
from std_msgs.msg import Float32, String
from rosserial_arduino.msg import Adc
from itertools import cycle

np.seterr('raise')

#global variables
speed = 0
regen = 0
state = ""
steering_angle = 1800
speed_odometry = 0

AUTONOMOUS_SPEED = 0 

NEUTRAL = 0
FORWARD = 1
REVERSE = 2

CAR_WIDTH = 1.5
CAR_LENGTH = 2.25

twothird = 850


class bcolors:
    HEADER = '\033[95m'     #mor
    OKBLUE = '\033[94m'     #mavi
    OKCYAN = '\033[96m'     #turkuaz
    OKGREEN = '\033[92m'    #yeşil
    WARNING = '\033[93m'    #sarı
    FAIL = '\033[91m'       #kırmızı
    ENDC = '\033[0m'        #beyaz
    BOLD = '\033[1m'        #kalın beyaz
    UNDERLINE = '\033[4m'   #aktı çizili beyaz


radius = None
def ackermann(fi, L, t):
    global radius
    toa = math.tan(math.radians(fi))
    radius = (L + (t/2 * toa)) / toa
    return radius 


def gen():
    gears = [NEUTRAL, FORWARD, NEUTRAL, REVERSE]    
    for i in cycle(gears):
        yield i

gear_generator = gen()

AUTONOMOUS = False
GEAR       = next(gear_generator)
DIRECTION  = True
LIGHTS     = False
EMERGENCY  = False
CURRENT    = 0


YAVIZ = False
SEKO  = False
TERMINATOR = False


# max_speed
YUSUF = 100

l_left_right = 0
l_up_down = 0
r_left_right = 0
r_up_down = 0
left_trigger = 1
right_trigger = 1
BUTTON_A = 0
BUTTON_B = 0
BUTTON_X = 0
BUTTON_Y = 0
BUTTON_LB = 0
BUTTON_RB = 0
BUTTON_BACK = 0
BUTTON_START = 0
BUTTON_STICK_LEFT = 0
BUTTON_STICK_RIGHT = 0 

# parking
is_parking_mode = False
is_durak_mode = False
park_coordinate = None
# 0->(dikey / x)
# 1->(yatay / y)
DIKEY = 0
YATAY = 1
park_distance = np.array([0, 0], dtype=np.float32)
#/parking

def mapper(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class PID(object):
    def __init__(self, kp, ki, kd):
        self.Kp 	  = kp
        self.Ki 	  = ki
        self.Kd 	  = kd
        self.p_error  = 0.0
        self.d_error  = 0.0
        self.i_error  = 0.0
        self.prev_cte = 0.0
        self.pidError = 0.0
        self.teta     = 0.0
        self.error    = 0.0
    
    def calculate(self, distance):
        self.error = distance - self.teta
        self.p_error = self.error
        self.i_error += self.error
        self.d_error = self.error - self.prev_cte
        self.prev_cte = self.error

        if self.error < 0.1:
            self.i_error = 0
        
        self.pidError = (self.p_error * self.Kp) + (self.i_error * self.Ki) + (self.d_error * self.Kd)

        self.teta += self.pidError

        if self.teta < -0.5:
            self.teta = -0.5
        elif self.teta > 0.5:
            self.teta = 0.5
        
        print("teta: {:.2f}".format(self.teta))
        return self.teta

def control_mahmut(data):
    # speed
    if (data.adc0 > 1000):
        data.adc0 = 1000
    elif (data.adc0 < 0):
        data.adc0 = 0
    # steer
    if (data.adc1 > 3600):
        data.adc1 = 3600
    elif (data.adc1 < 0):
        data.adc1 = 0
    # regen
    if (data.adc2 > 1000):
        data.adc2 = 1000
    elif (data.adc2 < 0):
        data.adc2 = 0


      
def lidar_data(veri_durak):
    global speed 
    global regen
    global state
    global steering_angle
    global is_parking_mode
    global is_durak_mode
    global park_coordinate
    global park_distance
    global speed_odometry

    sol_x = np.array(veri_durak.ranges[300:360])
    sol_y = np.array(veri_durak.ranges[240:300])
    sol_z = np.array(veri_durak.ranges[180:240])

    sag_x = np.array(veri_durak.ranges[1080:1140])
    sag_y = np.array(veri_durak.ranges[1140:1200])
    sag_z = np.array(veri_durak.ranges[1200:1260])

    on_array = np.array(np.concatenate((veri_durak.ranges[0:20], np.array(veri_durak.ranges[1419:1439]))))
    on_array[on_array > 25] = 25

    sol_array = (sol_x * 5 + sol_y * 3 + sol_z) / 9
    right_array = (sag_x * 5 + sag_y * 3 + sag_z) / 9

    sol_array[sol_array > 5] = 5
    right_array[right_array > 5] = 5 

    distances = {
        'right': np.average(right_array),
        'on' : np.average(on_array),
        'left': np.average(sol_array)
    }
    
    if TERMINATOR:
        if AUTONOMOUS:
            # Normal Autonomous
            if not is_parking_mode:
                speed = AUTONOMOUS_SPEED
                #
                #   CHECK THE DIRECTION
                #
                steering_angle = pid_controller.calculate(distances['left'] - distances['right'])
                angle = (steering_angle + 0.5) * 3600

                # doldur
                mahmut.adc0 = int(speed)
                mahmut.adc1 = int(angle)
                mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = 0

                yigit.adc0 = AUTONOMOUS_SPEED
                # doldur
            
            # <Parking Autonomous>
            elif is_parking_mode:
                speed = AUTONOMOUS_SPEED

                if True: # (park_distance[DIKEY] > 4):
                    print("Two Third Takip")                
                    if (twothird - park_coordinate) > 0:
                        target_diff = (twothird - park_coordinate) / twothird
                    else:
                        target_diff = (twothird - park_coordinate) / (1280 - twothird)
                else:
                    print("Middle Takip")
                    target_diff = (640 - park_coordinate) / 1280
                
                steer = (target_diff + 0.5) * 3600

                #test
                print("yatay", park_distance[YATAY])
                print("dikey", park_distance[DIKEY])
                if distances['on'] < 1 and speed_odometry >= 0:
                    speed = 0
                    regen = 1000
                    # brake

                # </Parking Autonomous>

                mahmut.adc0 = int(speed)
                mahmut.adc1 = int(steer)
                mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = 0
                
            elif is_durak_mode:
                pass
            
            #
            #   LOGGER
            # 
            if distances['left'] - distances['right'] > 0:
                d = "sol"
            elif distances['left'] - distances['right'] < 0:
                d = "sag"            
            else:
                d = "orta"

            if GEAR == 0:
                str_gear = "\tNEUTRAl"
            if GEAR == 1:
                str_gear = "\tFORWARD"
            if GEAR == 2:
                str_gear = "\tREVERSE"
            if is_parking_mode:
                str_gear += " (Parking Mode)"

            print(f"\n{str_gear}")
            print(f"dsrps: {d} error: {round(distances['left'] - distances['right'], 2)}")
            print(f"speed: {int(speed) :>5} regen: {int(regen) :>5}")
            
        # f710
        else:
            pid_controller.pidError = 0
            steering_angle = (l_left_right+1)*1800

            speed = mapper(right_trigger, 1, -1, 0, 1000)
            regen = mapper(left_trigger, 1, -1, 0, 1000)

            mahmut.adc0 = int(speed)           # speed (0, 1000)
            mahmut.adc1 = int(steering_angle)  # steering angle (0, 3600)
            mahmut.adc2 = int(regen)           # regen (0, 1000)
            mahmut.adc3 = int(GEAR)            # 
            mahmut.adc4 = int(AUTONOMOUS)      # autonomous
            mahmut.adc5 = int(EMERGENCY)       # emergency

            yigit.adc0 = mapper(speed, 0, 350, 0, 99)

            if distances['left'] - distances['right'] > 0.1:
                d = " left"
            elif distances['left'] - distances['right'] < -0.1:
                d = "right"            
            else:
                d = "  Mid"

            if GEAR == 0:
                str_gear = "\tNEUTRAl"
            if GEAR == 1:
                str_gear = "\tFORWARD"
            if GEAR == 2:
                str_gear = "\tREVERSE"

            print(f"\n{str_gear}")
            print(f"dsrps: {d} error: {round(distances['left'] - distances['right'], 2)}")
            print(f"speed: {int(speed) :>5} regen: {int(regen) :>5}")
    
    # no auth
    else:
        speed = 0
        mahmut.adc0 = int(0)
        mahmut.adc1 = int(1800)
        mahmut.adc2 = int(0)
        mahmut.adc3 = int(NEUTRAL)  
        mahmut.adc4 = int(0)        # autonomous
        mahmut.adc5 = int(0)        # emergency
        print(bcolors.FAIL + "!!! Authentication Required !!!" + bcolors.ENDC)

    yigit.adc0 = int(speed)
    yigit.adc1 = int(steering_angle)
    yigit.adc2 = int(regen)
    yigit.adc3 = int(GEAR)
    yigit.adc4 = int(AUTONOMOUS)
    yigit.adc5 = int(EMERGENCY)

    print("sag uzaklık:", np.average(right_array))
    print("sol uzaklık:", np.average(sol_array))
    lcd.publish(yigit)

    control_mahmut(mahmut)
    print(mahmut.adc1)
    arduino.publish(mahmut)



"""
    AXIS:
        0:  Left Right Stick Left
        1:  Up   Down  Stick Left
        2:  LT  (_start:=1, _end:=-1)
        3:  Left Right Stick Right
        4:  Up   Down  Stick Right
        5:  RT  (_start:=1, _end:=-1)
        6:  CrossKey (_left:= 1, _right:= -1)
        7:  CrossKey (_up:= 1, _down:= -1)

    BUTTON:
        0:  A
        1:  B
        2:  X
        3:  Y
        4:  LB
        5:  RB
        6:  BACK
        7:  START
        8:  LOGITECH BUTTON
        9:  BUTTON STICK LEFT
        10: BUTTON STICK RIGHT
"""

def F1_2020(russell):
    """
        rosrun joy joy_node
    """
    global YAVIZ
    global SEKO
    global TERMINATOR
    global speed
    global AUTONOMOUS
    global REVERSE
    global DIRECTION
    global EMERGENCY
    global GEAR
    global LIGHTS
    global l_left_right
    global l_up_down
    global r_left_right
    global r_up_down
    global left_trigger
    global right_trigger
    global BUTTON_A
    global BUTTON_B
    global BUTTON_X
    global BUTTON_Y
    global BUTTON_LB
    global BUTTON_RB
    global BUTTON_BACK
    global BUTTON_START
    global BUTTON_STICK_LEFT
    global BUTTON_STICK_RIGHT

    global is_parking_mode

    # left strick
    l_left_right = russell.axes[0]
    l_up_down = russell.axes[1]
    # right strick
    r_left_right = russell.axes[3]
    r_up_down = russell.axes[4]


    # Destroyer of worlds
    if BUTTON_START:
        TERMINATOR ^= True


    # ABXY Buttons
    BUTTON_A = russell.buttons[0]
    BUTTON_B = russell.buttons[1]
    BUTTON_X = russell.buttons[2]
    BUTTON_Y = russell.buttons[3]
    # RB LB 
    BUTTON_LB = russell.buttons[4]
    BUTTON_RB = russell.buttons[5]
    # START STOP BUTTONS
    BUTTON_BACK = russell.buttons[6]
    BUTTON_START = russell.buttons[7]
    # Stick Buttons
    BUTTON_STICK_LEFT = russell.buttons[9]
    BUTTON_STICK_RIGHT = russell.buttons[10]

    
    if TERMINATOR:
        #
        # DON'T TOUCH
        # SECURITY !!!
        #    
        if russell.axes[2] == -1 and russell.axes[5] == -1:
            YAVIZ = True
        
        if YAVIZ == True and russell.axes[2] == 1 and russell.axes[5] == 1:
            SEKO = True
        #
        # DON'T TOUCH
        # SECURITY !!!
        #

        # triggers
        if (YAVIZ == True) and (SEKO == True):
            left_trigger = russell.axes[2]
            right_trigger = russell.axes[5]
        else:
            # (1, -1) => (0, max_speed)
            left_trigger = 1
            right_trigger = 1
            
        if BUTTON_Y:
            AUTONOMOUS ^= True
        if BUTTON_X:
            GEAR = next(gear_generator)
        if BUTTON_A:
            is_parking_mode ^= True
        if BUTTON_B:
            EMERGENCY ^= True


"""
<PARK CALLBACKS>
"""

# String data
def yolo_callback(data):
    global state
    global is_parking_mode
    state = data.data
    
    if state == "Park Yeri":
        #is_parking_mode = True
        pass

# String data
def park_coordinate_callback(park_data):
    global park_coordinate
    if park_data.data=='None':
        park_coordinate='None'
    else:
        park_coordinate = float(park_data.data)


# Point(x, y, z) data
def park_distance_callback(data):
    park_distance[0] = abs(data.z)
    park_distance[1] = abs(data.y)

"""
</PARK CALLBACKS>
"""

def haydi_gel_icelim(data):
    global CURRENT 
    CURRENT = data.z


if __name__ == "__main__":
    rospy.init_node('pid',anonymous=True)
    rospy.Subscriber('/scan', LaserScan, lidar_data, queue_size=10)
    rospy.Subscriber('/joy', Joy, F1_2020, queue_size=10)
    rospy.Subscriber('/pot_topic', Vector3, haydi_gel_icelim, queue_size=10)
    rospy.Subscriber('yolo_park', String, park_coordinate_callback, queue_size=10)
    rospy.Subscriber('/zed_yolo_raw_distance', String, yolo_callback, queue_size=10)
    rospy.Subscriber('zed_yolo_sign_coord', Point, park_distance_callback, queue_size=10)

    arduino = rospy.Publisher("/seko", Adc, queue_size=10, latch=True)
    lcd = rospy.Publisher("/screen", Adc, queue_size=10, latch=True)

    f710 = Joy()
    mahmut = Adc()
    yigit = Adc()
    pid_controller = PID(0.8, 0.0075, 0.225)

    while not rospy.is_shutdown():
        rospy.spin()
