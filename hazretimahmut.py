#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from numpy.lib.function_base import average
import rospy
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import Vector3
import numpy as np
from std_msgs.msg import Float32


#global variables
lazer = []
speed = 30.0
state = ""
cnt_auto = 0
AUTONOMOUS = False

cnt_reverse = 0
REVERSE = False

MAX_SPEED = 100
DIRECTION = 1

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

        print("distance: {:.2f}".format(distance))
        print("teta: {:.2f}".format(self.teta))

        return self.teta

        
      
def lidar_data(veri_durak):
    global speed 
    """ 
    lazer = veri_durak
    
    sol_array = np.array(veri_durak.ranges[320:400])
    right_array = np.array(veri_durak.ranges[1040:1120])
    on_array = np.array(veri_durak.ranges[0:20] + np.array(veri_durak.ranges[1419:1439]))


    sol_array[sol_array > 5] = 5
    right_array[right_array > 5] = 5 

    distances = {
        'right': np.average(right_array),
        'on' : np.average(on_array) / 2,
        'left': np.average(sol_array)
    }
    """
    #if AUTONOMOUS:
    if False:
        #
        #   CHECK THE DIRECTION
        #
        steering_angle = pid_controller.calculate(distances['left'] - distances['right'])
        angle = (steering_angle + 0.5) * 3600

        yigit.x = speed
        yigit.y = angle
        yigit.z = 0

        print("ön", distances['on'])
        print("left", distances['left'])
        print("right", distances['right'])

        if distances['on'] < 5:
            speed = 0
            yigit.x = speed
            pid_controller.pidError = 0
        else:
            speed = 150
            yigit.x = speed
    else:
        pid_controller.pidError = 0
        steering_angle = (l_left_right+1)*1800

        speed = mapper(right_trigger, 1, -1, 0, 1000) * DIRECTION
        regen = mapper(left_trigger, 1, -1, 0, 1000)

        yigit.x = speed
        yigit.y = steering_angle
        yigit.z = regen

        #print("STEERING", round(steering_angle, 2), "SPEED", round(yigit.x,2), "Direction", DIRECTION)

    arduino.publish(yigit)


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
    global cnt_auto
    global AUTONOMOUS
    global cnt_reverse
    global REVERSE
    global DIRECTION
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

    cnt_auto += BUTTON_Y
    AUTONOMOUS = cnt_auto % 2

    if (DIRECTION == 1) and speed < 100:
        cnt_reverse += BUTTON_X
        REVERSE = cnt_reverse % 2
            
        if REVERSE:
            DIRECTION = -1
        else:
            DIRECTION = 1
    elif (DIRECTION == -1) and speed > -100:
        cnt_reverse += BUTTON_X
        REVERSE = cnt_reverse % 2
            
        if REVERSE:
            DIRECTION = -1
        else:
            DIRECTION = 1


""" def yolo_callback(data):
    global state
    state = data.data """


if __name__ == "__main__":
    rospy.init_node('pid',anonymous=True)
    rospy.Subscriber('/scan', LaserScan, lidar_data)
    rospy.Subscriber('/joy', Joy, F1_2020)
    #rospy.Subscriber('/yolo_topic', String, yolo_callback)

    arduino = rospy.Publisher("/seko", Vector3, queue_size=10, latch=True)

    yigit = Vector3()
    pid_controller = PID(0.8, 0.01, 0.225)
    f710 = Joy()

    rospy.spin()
    
    
    
