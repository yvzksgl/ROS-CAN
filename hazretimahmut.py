#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


from numpy.lib.function_base import average
import rospy
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import Vector3
import numpy as np
from std_msgs.msg import Float32
from rosserial_arduino.msg import Adc
from itertools import cycle

#global variables
speed = 30.0
state = ""

NEUTRAL = 0
FORWARD = 1
REVERSE = 2

def gen():
    gears = [NEUTRAL, FORWARD, NEUTRAL, REVERSE]    
    for i in cycle(gears):
        yield i

gear_generator = gen()

AUTONOMOUS = False
GEAR = FORWARD
DIRECTION = True
LIGHTS = False
EMERGENCY = False

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

        self.pidError = (self.p_error * self.Kp) + (self.i_error * self.Ki) + (self.d_error * self.Kd)

        self.teta += self.pidError

        if self.teta < -0.5:
            self.teta = -0.5
        elif self.teta > 0.5:
            self.teta = 0.5
        
        print("teta: {:.2f}".format(self.teta))
        return self.teta


      
def lidar_data(veri_durak):
    global speed 


    sol_array = np.array(veri_durak.ranges[320:400])
    right_array = np.array(veri_durak.ranges[1040:1120])
    on_array = np.array(veri_durak.ranges[0:20] + np.array(veri_durak.ranges[1419:1439]))

    on_array[on_array > 25] = 25

    sol_array[sol_array > 5] = 5
    right_array[right_array > 5] = 5 

    distances = {
        'right': np.average(right_array),
        'on' : np.average(on_array) / 2,
        'left': np.average(sol_array)
    }
    
    if AUTONOMOUS:
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
        # doldur

        print(f"{distances['left'] - distances['right']}")
        
        #if distances['on'] < 5:
        if False:
            mahmut.adc0 = 0
            pid_controller.pidError = 0
        else:
            speed = 150
            mahmut.adc0 = int(speed)
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

        print(f"{distances['left'] - distances['right']}")

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

    # left strick
    l_left_right = russell.axes[0]
    l_up_down = russell.axes[1]
    # right strick
    r_left_right = russell.axes[3]
    r_up_down = russell.axes[4]

    # triggers
    left_trigger = russell.axes[2]
    right_trigger = russell.axes[5]

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


    if BUTTON_Y:
        AUTONOMOUS ^= True
    if BUTTON_X:
        GEAR = next(gear_generator)
    if BUTTON_A:
        LIGHTS ^= True
    if BUTTON_B:
        EMERGENCY ^= True


""" def yolo_callback(data):
    global state
    state = data.data """


if __name__ == "__main__":
    rospy.init_node('pid',anonymous=True)
    rospy.Subscriber('/scan', LaserScan, lidar_data)
    rospy.Subscriber('/joy', Joy, F1_2020)
    #rospy.Subscriber('/yolo_topic', String, yolo_callback)

    arduino = rospy.Publisher("/seko", Adc, queue_size=10, latch=True)

    f710 = Joy()
    mahmut = Adc()
    pid_controller = PID(0.8, 0.0075, 0.225)

    while not rospy.is_shutdown():
        rospy.spin()
