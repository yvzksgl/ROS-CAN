#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#/*
#    @year:        2020/2021
#    @author:      Sekomer
#    @touch:       aksoz19@itu.edu.tr
#*/

import os
import sys
import csv
import math
import rospy
import numba
import seaborn
import numpy as np
from  matplotlib import pyplot as plt
from  rosserial_arduino.msg import Adc
from  std_msgs.msg import Float32, String
from  itertools import cycle, filterfalse
from  sensor_msgs.msg import LaserScan, Joy
from  numpy.lib.function_base import average
from  geometry_msgs.msg import Vector3, Point
from  pynput.keyboard import Key, Listener, GlobalHotKeys

np.seterr('raise')

BICYCLE_LENGTH = 3.5

sol_laser = None
sag_laser = None
collecting_data = False

#global variables
speed = 0
regen = 0
state = ""
steering_angle = 1800
speed_odometry = 0
brake_speed = 0
brake_motor_direction = 1

AUTONOMOUS_SPEED = 150

POT_CENTER = 1800
MAX_RPM_MODE_SPEED = 1000

RPM_MODE = 1
CURRENT_MODE = 0

driving_mode = RPM_MODE

DORU = (1 == 1)

left_tracking = False
right_tracking = False
mid_tracking = True

NEUTRAL = 0
FORWARD = 1
REVERSE = 2

CAR_WIDTH = 1.5
CAR_LENGTH = 2.25

twothird = 950

CURRENT    = 0
BUS_VOLTAGE = 0

CRUISE_CONTROL = False
cruiseSpeed = 50

YAVIZ = False
SEKO  = False
TERMINATOR = False


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


global_radius = None
@numba.jit(nopython=True, fastmath=True)
def ackermann(fi, L, t):
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

SOL_FIXED = 2
SAG_FIXED = 2
kararVerici = np.array([0, 1, 0])

#durak1
first_stop_counter = False
first_stop = False

#durak2
second_stop_counter = False
second_stop = False

arduino_odometry = {
    'speed': .0,
    'steering_angle': .0,
    'bus_voltage': .0,
    'bus_current': .0
}

zed_x = None
zed_y = None
denk_coef = None
closest_point = None

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
is_curve_parking = False
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


class Steering_Algorithms:
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

            if distance < 0.15:
                self.i_error = 0
            
            self.pidError = (self.p_error * self.Kp) + (self.i_error * self.Ki) + (self.d_error * self.Kd)
            self.teta += self.pidError

            if self.teta < -0.5:
                self.teta = -0.5
            elif self.teta > 0.5:
                self.teta = 0.5
            
            print("teta: {:.2f}".format(self.teta))
            return self.teta

    ##
    #  BASE
    # 
    class Pure_Pursuit_Controller():
        def __init__(self, bicycle_length, lookahead_distance, target_direction_angle):
            self.bicycle_length = bicycle_length
            self.lookahead_distance = lookahead_distance
            self.target_direction_angle = target_direction_angle
            self.steering = 0

        def target_finder(self, laser):
            right_point_distance = np.average(laser.ranges[1080:1260])
            left_point_distance = np.average(laser.ranges[180:360])

            print("distance")
            print(right_point_distance)
            print(left_point_distance)

            if(right_point_distance > 5):
                right_point_distance = 5

            if(left_point_distance > 5):
                left_point_distance = 5

            hipotenus = math.sqrt(math.pow(right_point_distance,2) + math.pow(left_point_distance,2))
            self.lookahead_distance = hipotenus / 2
            theta = math.degrees(math.atan(right_point_distance / left_point_distance))
            self.target_direction_angle = 90 - 2 * theta
            self.steering_calculator()

        def steering_calculator(self):
            formula = ((2 * self.bicycle_length * math.sin(math.radians(self.target_direction_angle))) / self.lookahead_distance)
            
            self.steering = (math.degrees(math.atan((formula))))
            
            print("steering")
            print(self.steering)
            print(self.steering / 180)

            if self.steering < -.5:
                self.steering = -.5
            elif self.steering > .5:
                self.steering = .5
 
#####################################################################################################################

# String data
class Parking(object):
    def __init__(self):
        self.coefficents = []
        self.steering_angle = 0
        self.CP = list()

    def find_curve(self, x, y, degree):
        self.coefficents = np.polyfit(x, y, degree)
        return self.coefficents

    def find_all_point_on_the_curve(self,coef,park_x,park_y):
        all_x = np.zeros(shape=(10,))
        all_y = np.linspace(0,park_y,10)
        
        for a,x in enumerate(all_y):
            tot = 0
            for i,j in enumerate(reversed(coef)):
                tot += (x **i) * j
            all_x[a] = tot

        for i, j in zip(all_x,all_y):
            self.CP.append(CurvePoint(i, j))

    def choose_closest_point(self,all_x,all_y,x,y):
        min_error = 1000
        for obj in self.CP:
            error_x = obj.x - x
            error_y = obj.y - y
            error = (error_x)**2 + (error_y)**2
            if error < min_error: return obj

    def differ_sdp(self,all_x,all_y,x,y):
        min_x = 100
        min_y = 100
        distance = (min_x-x)**2+(min_y-y)**2
        for i in all_x:
            for j in all_y:
                new_min = (i-x)**2 + (j-y)**2
                if new_min<distance:
                    min_x = i
                    min_y = j
        point = [min_x,min_y]
        return point

    def is_that_point(self,p1,p2):
        error_x = p1[0]-p2[0]
        error_y = p1[1]-p2[1]

        error = math.sqrt(pow(error_x, 2) + pow(error_y, 2))

        if error<0.25:
            return True
        else:
            return False

    def point2pointAngle(self,p1,p2):
        error_x = p1[0]-p2[0]
        error_y = p1[1]-p2[1]
        
        angle = math.degrees(math.tana(error_x/error_y))
        
        return angle

    def distance_calculator(self,p1,p2):
        distance_x = p1[0]-p2[0]
        distance_y = p1[1]-p2[1]

        distance = ((distance_x)**2+(distance_y)**2)**1/2

        return distance


class CurvePoint:
    def __init__(self, x, y, visited=False):
        self.x = x
        self.y = y
        self.visited = visited

#####################################################################################################################

@numba.jit(nopython=True, fastmath=True, cache=True)
def potingen_straße(desired, difference):
    return (desired + 0.5 + difference) * 3600

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

    global brake_speed
    global brake_motor_direction

    global left_tracking
    global right_tracking
    global mid_tracking
    global driving_mode
    global pürşit
    global is_curve_parking

    global zed_x
    global zed_y
    global denk_coef
    global closest_point
    ####################################
    # YAPAY ZEKA DUNYAYI ELE GECIRECEK #
    ####################################
    global arduino_odometry
    global sol_laser
    global sag_laser
    global collecting_data

    sol_laser = np.array(veri_durak.ranges[0:369], dtype=np.float32)
    sag_laser = np.array(veri_durak.ranges[1079:1439], dtype=np.float32)
    
    sol_laser[sol_laser>25] = 25
    sag_laser[sag_laser>25] = 25

    if collecting_data:
        writer.writerow([sol_laser, sag_laser, arduino_odometry['steering_angle']])
        print(bcolors.WARNING + "COLLECTING DATA" + bcolors.ENDC)
    #######
    # END #
    #######

    """ sol_x = np.array(veri_durak.ranges[300:360])
    sol_y = np.array(veri_durak.ranges[240:300])
    sol_z = np.array(veri_durak.ranges[180:240])

    sag_x = np.array(veri_durak.ranges[1080:1140])
    sag_y = np.array(veri_durak.ranges[1140:1200])
    sag_z = np.array(veri_durak.ranges[1200:1260]) """

    on_array = np.array(np.concatenate((veri_durak.ranges[0:20], np.array(veri_durak.ranges[1419:1439]))))

    #sol_array = (sol_x * 5 + 3 * sol_y + sol_z) / 9
    #right_array = (sag_x * 5 + 3 * sag_y + sag_z) / 9

    sol_array = np.array(veri_durak.ranges[320:360])
    right_array = np.array(veri_durak.ranges[1020:1080])

    sol_array[sol_array > 5] = 5
    right_array[right_array > 5] = 5

    distances = {
        'right': np.average(right_array),
        'on' : np.average(on_array),
        'left': np.average(sol_array)
    }
    
    
    if TERMINATOR:
        if AUTONOMOUS:
            print(bcolors.WARNING + "AUTONOMOUS" + bcolors.ENDC)
            driving_mode = RPM_MODE
            # Normal Autonomous
            if mid_tracking == True:
                print(bcolors.FAIL + "MID_TRACKING" + bcolors.ENDC)
                #
                #   CHECK THE DIRECTION

                #pürşit.target_finder(laser=veri_durak)
                right_point_distance = np.average(right_array)
                left_point_distance = np.average(sol_array)

                if(right_point_distance > 5):
                    right_point_distance = 5

                if(left_point_distance > 5):
                    left_point_distance = 5

                bicycle_length = 3.5
                target_direction_angle = 1
                lookahead_distance = 1

                hipotenus = math.sqrt(math.pow(right_point_distance,2) + math.pow(left_point_distance,2))
                lookahead_distance = hipotenus / 2
                theta = math.degrees(math.atan(right_point_distance / left_point_distance))
                target_direction_angle = 90 - 2 * theta

                formula = ((2 * bicycle_length * math.sin(math.radians(target_direction_angle))) / lookahead_distance)
                steering = (math.degrees(math.atan((formula)))) / 180

                steering = fast_pp(right_point_distance, left_point_distance)

                #pid method
                #steering = -pid_controller.calculate(distances['left'] - distances['right'])
    
                if steering < -.5:
                    steering = -.5
                elif steering > .5:
                    steering = .5
                
                angle = potingen_straße(steering, POT_CENTER-1800)
                print("desired direction angle:", angle)

                # doldur
                mahmut.adc0 = int(AUTONOMOUS_SPEED)
                mahmut.adc1 = int(angle)
                mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = int(driving_mode)
            
            elif left_tracking == DORU:
                #
                #   CHECK THE DIRECTION
                #
                steering_angle = pid_controller.calculate(distances['left'] - SAG_FIXED)
                angle = (steering_angle + 0.5) * 3600

                # doldur
                mahmut.adc0 = int(AUTONOMOUS_SPEED)
                mahmut.adc1 = int(angle)
                #mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = int(driving_mode)

            elif right_tracking == DORU:
                #
                #   CHECK THE DIRECTION
                #
                steering_angle = pid_controller.calculate(SOL_FIXED - distances['right'])
                angle = (steering_angle + 0.5) * 3600

                # doldur
                mahmut.adc0 = int(AUTONOMOUS_SPEED)
                mahmut.adc1 = int(angle)
                #mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = int(driving_mode)

            # <Parking Autonomous>
            elif is_parking_mode:
                speed = AUTONOMOUS_SPEED

                if (park_distance[YATAY] > 2):
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
                #mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = int(driving_mode)
                

            #############################################
            elif is_curve_parking:
                speed = AUTONOMOUS_SPEED

                if park.is_that_point(closest_point, [zed_x, zed_y]):
                    closest_point = park.choose_closest_point([zed_x,zed_y])

                bicycle_length = 3.5
                target_direction_angle = 1
                lookahead_distance = 1

                distance = math.sqrt(math.pow(closest_point[0]-zed_x,2) + math.pow(closest_point[1]-zed_y,2))
                lookahead_distance = distance 
                
                theta =  park.point2pointAngle(closest_point,[zed_x,zed_y])
                target_direction_angle = 90 - theta

                formula = ((2 * bicycle_length * math.sin(math.radians(target_direction_angle))) / lookahead_distance)
                steering = (math.degrees(math.atan((formula)))) / 180
                
                print("steering")
                print(steering)

                if steering < -.5:
                    steering = -.5
                elif steering > .5:
                    steering = .5

                mahmut.adc0 = int(speed)
                mahmut.adc1 = int(steering)
                #mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = int(driving_mode)
            #############################################

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
            steering_angle = (l_left_right+1)*1800  # 0 3600

            if not CRUISE_CONTROL and driving_mode == CURRENT_MODE:
                speed = mapper(right_trigger, 1, -1, 0, 1000)
            elif not CRUISE_CONTROL and driving_mode == RPM_MODE:
                speed = mapper(right_trigger, 1, -1, 0, MAX_RPM_MODE_SPEED)            
            
            regen = mapper(left_trigger, 1, -1, 0, 1000)

            mahmut.adc0 = int(speed)           # speed (0, 1000)
            mahmut.adc1 = int(steering_angle)  # steering angle (0, 3600)
            mahmut.adc2 = int(regen)           # regen (0, 1000)
            mahmut.adc3 = int(GEAR)            # 
            mahmut.adc4 = int(AUTONOMOUS)      # autonomous
            mahmut.adc5 = int(driving_mode)    # mode

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
        mahmut.adc3 = int(GEAR)  
        mahmut.adc4 = int(0)        # autonomous
        mahmut.adc5 = int(0)        # emergency
        print(bcolors.FAIL + "!!! Authentication Required !!!" + bcolors.ENDC)

    if driving_mode == 1:
        print("Driving Mode: RPM")
    elif driving_mode == 0:
        print("Driving Mode: CURRENT")
    else:
        print("Sen şu an bir şeyleri yedin")

    print("sag uzaklık:", np.average(right_array))
    print("sol uzaklık:", np.average(sol_array))

    control_mahmut(mahmut)
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
    global AUTONOMOUS_SPEED
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
    global is_curve_parking

    global left_tracking
    global right_tracking
    global mid_tracking
    global driving_mode
    global brake_motor_direction

    global collecting_data
    global CRUISE_CONTROL

    # left strick
    l_left_right = russell.axes[0]
    l_up_down = russell.axes[1]
    # right strick
    r_left_right = russell.axes[3]
    r_up_down = russell.axes[4]


    # Destroyer of worlds
    if BUTTON_START:
        TERMINATOR ^= True
        YAVIZ = SEKO = False


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
        
        if YAVIZ and SEKO and BUTTON_BACK:
            CRUISE_CONTROL ^= True
            cruiseSpeed = 50

        if BUTTON_Y:
            AUTONOMOUS ^= True
        if BUTTON_X and right_trigger == 1 and speed == 0:
            GEAR = next(gear_generator)
        if BUTTON_A:
            driving_mode ^= True
        if BUTTON_B:
            collecting_data ^= True
        
        if BUTTON_LB:
            if AUTONOMOUS:
                AUTONOMOUS_SPEED -= 25
            elif CRUISE_CONTROL:
                speed -= 25
            else:
                ...
        if BUTTON_RB:
            if AUTONOMOUS:
                AUTONOMOUS_SPEED += 25
            elif CRUISE_CONTROL:
                speed += 25
            else:
                ...
        if BUTTON_STICK_RIGHT:
            is_curve_parking ^= True
            mid_tracking ^= True
"""
<PARK CALLBACKS>
"""


SOL = 0
ORTA = 1
SAG = 2
# String data
# @sikinti
def yolo_callback(data):
    global speed
    global state
    global is_parking_mode
    global kararVerici
    global left_tracking
    global right_tracking
    global mid_tracking

    #durak
    global first_stop_counter
    global first_stop
    
    #state = data.data

    #tabela1,distance1;tabela2,distance2
    
    if data.data != " 0":
        datas = data.data.split(';')

        imitasyon_sol = None
        imitasyon_sag = None
        imitasyon_ort = None

        r_u_sure = False

        for tabela in datas:
            label, distance = tabela.split(',')
            #########################pARk###########################
            if label == "Park Yeri" and distance < 10:
                #is_parking_mode = True
                pass
            elif label == "Park Yapilmaz" and distance < 15:
                left_tracking = True
                right_tracking = False
                mid_tracking = False
            ###############################################################
            
            elif label == "sola donulmez":
                kararVerici[SOL]  = None
                kararVerici[SAG]  = None
                kararVerici[ORTA]  = None
            elif label == "saga donulmez":
                kararVerici[SOL]  = None
                kararVerici[SAG]  = None
                kararVerici[ORTA]  = None
            elif label == "Girilmez":
                kararVerici[SOL]  = None
                kararVerici[SAG]  = None
                kararVerici[ORTA]  = None
            ###############################################################
            elif label == "ileriden sola mecburi yon" and distance < 3.5:
                left_tracking = True
                right_tracking = False
                mid_tracking = False
                break
            elif label == "ileriden saga mecburi yon" and distance < 3.5:
                left_tracking = False
                right_tracking = True
                mid_tracking = False
                break
            ###############################################################
            elif label == "Durak" and distance < 2.5:
                first_stop = True
                
                if first_stop_counter > 30:
                    started_moving_first = True
                r_u_sure = True
                continue
            ###############################################################
            elif label == "yesil isik":
                speed = AUTONOMOUS_SPEED
                regen = 0
                r_u_sure = True
                continue
            elif label == "kirmizi isik" and distance < 2.0:
                speed = 0
                regen = 1000
                r_u_sure = True
                continue
            ###############################################################
            elif label == "Sola Mecburi Yön" and distance < 3.5:
                left_tracking = True
                right_tracking = False
                mid_tracking = False
                break
            elif label == "Saga Mecburi Yön" and distance < 3.5:
                right_tracking = True
                left_tracking = False
                mid_tracking = False
                break
            else:
                right_tracking = False
                left_tracking = False
                mid_tracking = True
        

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

def zed_pose(data):
    zed_x = data.z
    zed_y = data.x
"""
</PARK CALLBACKS>
"""

""" Keyboard """
def on_press(key):
    global mahmut
    global hiz
    global direk
    global arduino
    global GEAR

    if(key == Key.up):
        hiz += 25
        GEAR = 1
    elif(key == Key.down):
        hiz -= 25
        GEAR = 2
    elif(key == Key.left):
        direk = 3600
    elif(key == Key.right):
        direk = 0
    
    if hiz < 0:
        GEAR = 2
    if hiz >= 0:
        GEAR = 1

def on_release(key):
    global mahmut
    global hiz
    global direk
    global arduino
    global GEAR

    if(key == Key.left):
        direk = 1800
    elif(key == Key.right):
        direk = 1800
""" /Keyboard """

"""
<Ardu Odom>
"""
def haydi_gel_icelim(data):
    global arduino_odometry 
    
    arduino_odometry['speed'] = data.adc0
    arduino_odometry['steering_angle'] = data.adc1
    arduino_odometry['bus_voltage'] = data.adc2
    arduino_odometry['bus_current'] = data.adc3
"""
</Ardu Odom>
"""

@numba.jit(nopython=True, fastmath=True, cache=True)
def fast_pp(right_point_distance, left_point_distance, bicycle_length = BICYCLE_LENGTH):
    if(right_point_distance > 5):
        right_point_distance = 5

    if(left_point_distance > 5):
        left_point_distance = 5

    hipotenus = math.sqrt(math.pow(right_point_distance,2) + math.pow(left_point_distance,2))
    lookahead_distance = hipotenus / 2
    theta = math.degrees(math.atan(right_point_distance / left_point_distance))
    target_direction_angle = 90 - (2 * theta)

    formula = (2 * bicycle_length * math.sin(math.radians(target_direction_angle))) / lookahead_distance
    return math.degrees(math.atan(formula)) / 180

if __name__ == "__main__":
    file = open("çöp.csv", "a+")
    writer = csv.writer(file)
    writer.writerow(["SOL", "SAG", "DIREKSIYON"])

    # just in time compiler #
    potingen_straße(31, 31)
    ackermann(31, 31, 31)
    fast_pp(31, 31)

    rospy.init_node('mahmut',anonymous=True)
    rospy.Subscriber('/scan', LaserScan, lidar_data, queue_size=10)
    rospy.Subscriber('/joy', Joy, F1_2020, queue_size=10)
    rospy.Subscriber('/pot_topic', Adc, haydi_gel_icelim, queue_size=10)
    rospy.Subscriber('yolo_park', String, park_coordinate_callback, queue_size=10)
    rospy.Subscriber('/zed_yolo_raw_distance', String, yolo_callback, queue_size=10)
    rospy.Subscriber('zed_yolo_sign_coord', Point, park_distance_callback, queue_size=10)
    rospy.Subscriber('zed_yolo_pose',Point,zed_pose,queue_size = 10)

    arduino = rospy.Publisher("/seko", Adc, queue_size=10, latch=True)
    lcd = rospy.Publisher("/screen", Adc, queue_size=10, latch=True)

    f710 = Joy()
    mahmut = Adc()

    # driving with keyboard    
    hiz = 0
    direk = 1800
    keyboard_listener = Listener(on_press=on_press, on_release=on_release)
    #keyboard_listener.start()


    pid_controller = Steering_Algorithms.PID(0.8, 0.0075, 0.225)
    pürşit = Steering_Algorithms.Pure_Pursuit_Controller(5, 5, 1)
    park = Parking()

    #x_coords = [0,park_distance[0],park_distance[0]]
    #y_coords = [0,park_distance[1],park_distance[1]]

    #denk_coef = park.find_curve(x_coords,y_coords,2)
    #park.find_all_point_on_the_curve(denk_coef,park_distance[0],park_distance[1])
    #closest_point = park.choose_closest_point(zed_x,zed_y)
    
    while not rospy.is_shutdown():
        rospy.spin()

    file.close()


