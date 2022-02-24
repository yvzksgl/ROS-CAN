#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#/*
#    @year:        2020/2021
#    @authors:     Sekomer, ubuntumevuz
#    @touch:       aksoz19@itu.edu.tr
#    @touch:       koseoglumu18@itu.edu.tr
#*/

import os
import sys
import csv
import math
import time
import rospy
import numba
import seaborn
import numpy as np
from   numba import float32 as f32
from   matplotlib import pyplot as plt
from   rosserial_arduino.msg import Adc
from   std_msgs.msg import String
from   itertools import cycle
from   sensor_msgs.msg import LaserScan, Joy
from   numpy.lib.function_base import average
from   geometry_msgs.msg import Point
#from  pynput.keyboard import Key, Listener

np.seterr('raise')

# global variables #
TWO_THIRD_PARKING = True
REVERSE_PARKING = False
speed = 0
regen = 0
DURMAK = 0
steering_angle = 1800
speed_odometry = 0
brake_speed = 0
brake_motor_direction = 1
sol_laser = None
sag_laser = None
collecting_data = False
SOL = 0
ORTA = 1
SAG = 2 
AUTONOMOUS_SPEED = 39 # never change this variable randomly!!!
AUTONOMOUS_SPEED_RECOVERY = AUTONOMOUS_SPEED
POT_CENTER = 1800
MAX_RPM_MODE_SPEED = 200
RPM_MODE = 1
CURRENT_MODE = 0
driving_mode = RPM_MODE
DORU = (1 == 1)

left_tracking = 0
right_tracking = 1
mid_tracking = 0

girildim_counter = 0
girildim_bool = True

NEUTRAL = 0
FORWARD = 1
REVERSE = 2
CAR_WIDTH = 1.5
CAR_LENGTH = 2.25
CURRENT    = 0
BUS_VOLTAGE = 0
BICYCLE_LENGTH = 3.5
SOL_FIXED = 2.65
SAG_FIXED = 2.65
kararVerici = np.array([0, 1, 0])
recently_stopped = False
recently_stopped_kirmizi = False
orhandaldal = np.inf
mahmut_tuncer = np.inf
brake = False
brake_value = 0
roswtf = False
FULL_RIGHT = 0
FULL_LEFT = 3600
left_point_distance = 0
right_point_distance = 0
karar_verici_yakın_zamanda_calisti = False
kirmizida_dur_lan = False
ahaburasıdaboşmuşıheahıeah = False
olceriz_sıkıntı_yog = np.inf
YOLCU = 37
hazreticounter = 0
experimental_park_stage_1 = False
experimental_park_stage_2 = False
mahmutapozisyonçeşitliliği = np.array([.0, .0, .0])
igotthepose = False
yesil_gorundu = False

büllük = False
güllük = np.array([.0,.0,.0])

# durak experimental #
first_stop_counter = False
first_stop = False
second_stop_counter = False
second_stop = False
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#

arduino_odometry = {
    'speed': .0,
    'steering_angle': .0,
    'bus_voltage': .0,
    'bus_current': .0
}

# terminal colors #
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
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#

# controller #
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
CRUISE_CONTROL = False
YAVIZ = False
SEKO  = False
TERMINATOR = False
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

stage1 = True
stage2 = False
stage3 = False
stage4 = False

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#

# parking
is_parking_mode = False
is_curve_parking = False
is_durak_mode = False
park_coordinate = None
zed_x = None
zed_y = None
zed_z = None
durak_dik_start = None
denk_coef = None
closest_point = None
is_curve_created = False
CRITICAL_PARKING_DISTANCE = 15
CALCULATE_PARKING_SIGN_DISTANCE = 10
locked_on_target = False
parking_sign_current_distance = None
park_left_not_started = True

ACKERMAN_RADIUS = 4
LABEL_OFFSET = 9.
ERROR_ACCEPTANCE = .3

karsiya_geciyorum = False

DIKEY = 0 # 0->(dikey / x)
YATAY = 1 # 1->(yatay / y)
twothird = 1400
park_distance = np.inf
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#

# helper funcs #
global_radius = None
@numba.jit(f32(f32, f32, f32), nopython=True, fastmath=True)
def ackermann(fi, L, t):
    toa = math.tan(math.radians(fi))
    radius = (L + (t/2 * toa)) / toa
    return radius 

@numba.jit(f32(f32, f32, f32, f32, f32), nopython=True, fastmath=True, cache=True)
def mapper(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

@numba.jit(f32(f32, f32), nopython=True, fastmath=True, cache=True)
def potingen_straße(desired, difference):
    return (desired + 0.5 + difference) * 3600

@numba.jit(f32(f32, f32, f32, f32, f32, f32), nopython=True, fastmath=True, cache=True)
def fast_pid(p_error, Kp, i_error, Ki, d_error, Kd):
    return (p_error * Kp) + (i_error * Ki) + (d_error * Kd)

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

@numba.jit(nopython=True, fastmath=True, cache=True)
def fast_pp2(right_point_distance, left_point_distance, bicycle_length = BICYCLE_LENGTH):
    hipotenus = math.sqrt(math.pow(right_point_distance,2) + math.pow(left_point_distance,2))
    lookahead_distance = hipotenus / 2
    theta = math.degrees(math.atan(right_point_distance / left_point_distance))
    target_direction_angle = 90 - (2 * theta)

    formula = (2 * bicycle_length * math.sin(math.radians(target_direction_angle))) / lookahead_distance
    return math.degrees(math.atan(formula)) / 180


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
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#

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
            
            self.pidError = fast_pid(self.p_error, self.Kp, self.i_error, self.Ki, self.d_error, self.Kd)
            self.teta += self.pidError

            if self.teta < -0.5:
                self.teta = -0.5
            elif self.teta > 0.5:
                self.teta = 0.5
            
            print("teta: {:.2f}".format(self.teta))
            return self.teta


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
"""
    @future
    Requires positional knowledge
"""
class park_seko(object):
    def __init__(self, ackermann_radius, label_offset, error_acceptance = 0.5):
        self.__target = []
        self.__margin = np.inf
        self.__destiny = False
        self.__current_pos = []
        self.__label_offset = label_offset
        self.__error_acceptance = error_acceptance
        self.__ackermann_radius = ackermann_radius

    def __eval__(self):
        # offset x mi y mi kontrol et
        #@ZED değiştirilecek!
        ackermann_target_x = self.__target[0] - ACKERMAN_RADIUS - LABEL_OFFSET # @!!!
        ackermann_target_y = self.__target[1] - ACKERMAN_RADIUS # @!!!
        xd = math.pow(ackermann_target_x - self.__current_pos[0], 2)
        yd = math.pow(ackermann_target_y - self.__current_pos[1], 2)
        self.__margin = math.sqrt(xd + yd)
        
        if self.__margin < self.__error_acceptance:
            self.__destiny = True


"""
    Deprecated
"""
class serhatos(object):
    def __init__(self):
        self.coefficents = []
        self.steering_angle = 0
        self.CP = list()
    def find_curve(self, x, y, degree):
        self.coefficents = np.polyfit(x, y, degree)
        return self.coefficents
    def find_all_point_on_the_curve(self,coef,park_y):
        all_y = np.zeros(shape=(10,))
        all_x = np.linspace(-3,park_y,10)
        for a,x in enumerate(all_x):
            tot = 0
            for i,j in enumerate(reversed(coef)):
                tot += (x **i) * j
            all_y[a] = tot
        for i, j in zip(all_x,all_y):
            #self.CP.append(CurvePoint(i, j))
            print(i, j)
    def choose_closest_point(self, x, y):
        min_error = np.inf
        return_obj = None
        for obj in self.CP:
            if obj.visited: continue
            error_x = obj.x - x
            error_y = obj.y - y
            error = math.pow(error_x, 2) + math.pow(error_y, 2)
            if error < min_error: 
                return_obj = obj
                min_error = error
        return return_obj
    def is_that_point(self,p1,p2_x,p2_y):
        if not p1 or not p2_x: return
        error_x = p1.x-p2_x
        error_y = p1.y-p2_y
        error = math.sqrt(pow(error_x, 2) + pow(error_y, 2))
        if error<5:
            return True
        else:
            return False
    def point2pointAngle(self,p1,p2):
        error_x = p1.x-p2[0]
        error_y = p1.y-p2[1]
        angle = math.degrees(math.atan(error_x/error_y))
        return angle
    def distance_calculator(self,p1,p2):
        distance_x = p1[0]-p2[0]
        distance_y = p1[1]-p2[1]
        distance = ((distance_x)**2+(distance_y)**2)**1/2
        return distance
#####################################################################################################################

düz_gitmek = True
mid_start = np.array([.0, .0, .0], dtype=np.float32)

# main function #
def lidar_data(veri_durak):
    global YOLCU
    global DURMAK
    global speed 
    global regen
    global steering_angle
    global is_parking_mode
    global is_durak_mode
    global park_coordinate
    global speed_odometry
    global AUTONOMOUS_SPEED
    global FULL_RIGHT
    global brake_speed
    global brake_motor_direction
    global stage1
    global left_tracking
    global right_tracking
    global mid_tracking
    global driving_mode
    global pürşit
    global is_curve_parking
    # PARK #
    global zed_x
    global zed_y
    global zed_z
    global park_distance
    global durak_dik_start
    global denk_coef
    global closest_point
    global recently_stopped
    ####################################
    # YAPAY ZEKA DUNYAYI ELE GECIRECEK #
    ####################################
    global arduino_odometry
    global sol_laser
    global sag_laser
    global collecting_data
    global brake
    global brake_value
    global roswtf
    global orhandaldal
    global recently_stopped_kirmizi
    global mahmut_tuncer
    global parking_sign_current_distance
    global right_point_distance
    global left_point_distance
    global kirmizida_dur_lan
    global karar_verici_yakın_zamanda_calisti
    global ahaburasıdaboşmuşıheahıeah
    global olceriz_sıkıntı_yog
    global sol_array
    global right_array
    global hazreticounter

    global güllük
    global büllük

    global mahmutapozisyonçeşitliliği
    global igotthepose

    global TWO_THIRD_PARKING
    global REVERSE_PARKING

    global stage1
    global stage2
    global stage3
    global stage4

    global experimental_park_stage_1
    global experimental_park_stage_2

    global yesil_gorundu
    global kirmizi_distance

    global mid_start # array
    global düz_gitmek # bool

    global girildim_counter
    global girildim_bool

    global karsiya_geciyorum

    #sol_laser = np.array(veri_durak.ranges[0:369], dtype=np.float32)
    #sag_laser = np.array(veri_durak.ranges[1079:1439], dtype=np.float32)
    
    #sol_laser[sol_laser>25] = 25
    #sag_laser[sag_laser>25] = 25

    print("trackings:", left_tracking, mid_tracking, right_tracking)

    if ahaburasıdaboşmuşıheahıeah:
        pass
        # TUNE ET
        """
        while olceriz_sıkıntı_yog + 3 > time.time():
            mahmut.adc0 = AUTONOMOUS_SPEED
            mahmut.adc1 = 1800
            mahmut.adc2 = 0
            mahmut.adc3 = FORWARD
            mahmut.adc4 = True
            mahmut.adc5 = driving_mode 
        """
        while olceriz_sıkıntı_yog + 4 > time.time():
            print("ahaburası da bosmus yav")
            zol = np.array(veri_durak.ranges[560:672])
            zag = np.array(veri_durak.ranges[224:336])

            solumsu = zol[zol > 2.78] = 2.78
            sagimsi = zag[zag > 2.78] = 2.78

            xd = fast_pp(average(sagimsi), average(solumsu))
            
            mahmut.adc0 = AUTONOMOUS_SPEED
            mahmut.adc1 = int(xd)
            mahmut.adc2 = 0
            mahmut.adc3 = FORWARD
            mahmut.adc4 = True
            mahmut.adc5 = driving_mode
            time.sleep(1/20)
            arduino.publish(mahmut)
        
        ahaburasıdaboşmuşıheahıeah = False


    """
    if roswtf:
        mahmut.adc0 = 0
        mahmut.adc2 = 11000
        arduino.publish(mahmut)

        print(bcolors.WARNING+"Hoş geldiğiz!"+bcolors.ENDC)
        start = time.time()
        while time.time() - start < 10:
            arduino.publish(mahmut) 

        mahmut.adc0 = AUTONOMOUS_SPEED_RECOVERY
        mahmut.adc2 = 0        
        roswtf = False
        orhandaldal = time.time()
        print(bcolors.WARNING+"BBBBBB!"+bcolors.ENDC)
    """

    if orhandaldal + 15 < time.time():
        recently_stopped = False

    #if roswtf and abs(math.sqrt(math.pow(durak_dik_start[0] - zed_x, 2) + math.pow(durak_dik_start[1] - zed_z, 2))) > 4:
    if roswtf:
        print(bcolors.WARNING+"DURAK START"+bcolors.ENDC)
        start = time.time()

        if not igotthepose:
            mahmutapozisyonçeşitliliği[0] = zed_x
            mahmutapozisyonçeşitliliği[1] = zed_y
            mahmutapozisyonçeşitliliği[2] = zed_z
            igotthepose = True

        print("duraktan beri gittiğim yol: ", math.sqrt(math.pow(mahmutapozisyonçeşitliliği[0] - zed_x, 2) + math.pow(mahmutapozisyonçeşitliliği[2] - zed_z, 2)))
        # pose difference
        if math.sqrt(math.pow(mahmutapozisyonçeşitliliği[0] - zed_x, 2) + math.pow(mahmutapozisyonçeşitliliği[2] - zed_z, 2)) > 3.5:
            while time.time() < start + YOLCU:
                mahmut.adc0 = 0
                mahmut.adc1 = 1800
                mahmut.adc2 = 11000
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = driving_mode
                arduino.publish(mahmut)
                qoıwheqdw = round(time.time() - start, 2)
                print(bcolors.WARNING + f"{qoıwheqdw} saniyedir DURAKTAYIZZZZ" + bcolors.ENDC)
                time.sleep(1/20)

            orhandaldal = time.time()
            roswtf = False
            igotthepose = False
            left_tracking = True
            right_tracking = False
            mid_tracking = False
            print(bcolors.WARNING+"DURAK FINISH!"+bcolors.ENDC)


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

    on_array = np.array(veri_durak.ranges[440:456])
    sol_array = np.array(veri_durak.ranges[560:672])
    right_array = np.array(veri_durak.ranges[224:336])

    sol_array[sol_array > 5] = 5
    right_array[right_array > 5] = 5

    distances = {
        'right': np.average(right_array),
        'left': np.average(sol_array),
        'on' : np.average(on_array)
    }

    print(bcolors.WARNING + "ON MESAFE:    " + bcolors.ENDC, distances['on'])
    print(bcolors.WARNING + "park distance:" + bcolors.ENDC, park_distance)
    print(bcolors.WARNING + "park coord:   " + bcolors.ENDC, park_coordinate)
    
    print("GIRILMEZ COUNTER", girildim_counter)

    if TERMINATOR:
        if AUTONOMOUS:
            print(bcolors.WARNING + "AUTONOMOUS" + bcolors.ENDC)
            driving_mode = RPM_MODE

            if kirmizida_dur_lan:
                print(bcolors.WARNING + "KIRMIZIYA GELDİM SANKİ" + bcolors.ENDC)

                mahmut.adc0 = AUTONOMOUS_SPEED
                mahmut.adc1 = int(1800)
                mahmut.adc2 = 11000
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = driving_mode


            elif mid_tracking == True:
                if düz_gitmek:
                    mid_start[0] = zed_x
                    mid_start[1] = zed_y
                    mid_start[2] = zed_z

                    düz_gitmek = False

                print(bcolors.FAIL + "MID_TRACKING" + bcolors.ENDC)
                #
                #   CHECK THE DIRECTION

                #pürşit.target_finder(laser=veri_durak)

                mid_sol_array = np.array(veri_durak.ranges[399:435])
                mid_sag_array = np.array(veri_durak.ranges[462:498])


                right_point_distance = np.average(mid_sol_array)
                left_point_distance = np.average(mid_sag_array)

                if(right_point_distance > 25):
                    right_point_distance = 25

                if(left_point_distance > 25):
                    left_point_distance = 25

                print("right point distance", right_point_distance)
                print("left point distance", left_point_distance)
                
                steering = fast_pp2(right_point_distance, left_point_distance)
                print("steering:", steering)

                #pid method
                #steering = pid_controller.calculate(distances['left'] - distances['right'])
    
                if steering < -.5:
                    steering = -.5
                elif steering > .5:
                    steering = .5
                
                angle = potingen_straße(steering, POT_CENTER-1800)
                print("aci:", angle)
                print("desired direction angle:", angle)

                # doldur
                mahmut.adc0 = int(AUTONOMOUS_SPEED)
                mahmut.adc1 = int(angle)
                mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = int(RPM_MODE)

                right_point_distance = np.average(right_array)
                left_point_distance = np.average(sol_array)

                anlık_fark = math.sqrt(math.pow(mid_start[0] - zed_x, 2) + math.pow(mid_start[2] - zed_z, 2))

                print("ANLIKKKKK FARKKKK:::::", anlık_fark)
                print("GIRLDIM BOOl", girildim_bool)

                if right_point_distance < 3.0 and anlık_fark > 7:
                    left_tracking = True
                    mid_tracking = False
                    right_tracking = False
                    düz_gitmek = True
                    print(bcolors.FAIL + "ARANIYORUM HER YERDE KIRMIZI BULTENLE" + bcolors.ENDC)
                    girildim_bool = True
                    girildim_counter += 1
                    karsiya_geciyorum = False
                elif left_point_distance < 3.0 and anlık_fark > 7:
                    left_tracking = True
                    mid_tracking = False
                    right_tracking = False
                    düz_gitmek = True
                    print(bcolors.FAIL + "ARANIYORUM HER YERDE KIRMIZI BULTENLE" + bcolors.ENDC)
                    girildim_bool = True
                    girildim_counter += 1
                    karsiya_geciyorum = False
            
            elif left_tracking == DORU:
                print(bcolors.FAIL + "LEFT_TRACKING" + bcolors.ENDC)    #
                #   CHECK THE DIRECTION
                #

                right_point_distance = np.average(right_array)
                left_point_distance = np.average(sol_array)

                if(right_point_distance > 5):
                    right_point_distance = 5

                if(left_point_distance > 5):
                    left_point_distance = 5

                steering = fast_pp(SAG_FIXED, left_point_distance)

                #pid method
                #steering = pid_controller.calculate(distances['left'] - distances['right'])
    
                if steering < -.5:
                    steering = -.5
                elif steering > .5:
                    steering = .5
                
                angle = potingen_straße(steering, POT_CENTER-1800)

                # doldur
                mahmut.adc0 = int(AUTONOMOUS_SPEED)
                mahmut.adc1 = int(angle)
                mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = int(driving_mode)

            elif right_tracking == DORU:
                print(bcolors.FAIL + "RIGHT_TRACKING" + bcolors.ENDC)

                right_point_distance = np.average(right_array)
                left_point_distance = np.average(sol_array)

                if(right_point_distance > 5):
                    right_point_distance = 5

                if(left_point_distance > 5):
                    left_point_distance = 5

                steering = fast_pp(right_point_distance, SOL_FIXED)

                #pid method
                #steering = pid_controller.calculate(distances['left'] - distances['right'])
    
                if steering < -.5:
                    steering = -.5
                elif steering > .5:
                    steering = .5
                
                angle = potingen_straße(steering, POT_CENTER-1800)

                # doldur
                mahmut.adc0 = int(AUTONOMOUS_SPEED)
                mahmut.adc1 = int(angle)
                mahmut.adc2 = 0
                mahmut.adc3 = FORWARD
                mahmut.adc4 = True
                mahmut.adc5 = int(driving_mode)

            # <Parking Autonomous>
            elif is_parking_mode: #elif is_parking_mode:
                print(bcolors.WARNING + 10 * '-' + "PARKING MODE \n" + 10 * '-' + bcolors.ENDC)
                # full sag, ortalanınca middle takip park modu
                if 0:
                    if park_distance > 10 and not experimental_park_stage_1:
                        right_point_distance = np.average(right_array)
                        left_point_distance = np.average(sol_array)

                        if(right_point_distance > 5):
                            right_point_distance = 5
                        if(left_point_distance > 5):
                            left_point_distance = 5
                        
                        # !!!!!!!! 1.5 !!!!!!!!!!
                        steering = fast_pp(1.5 , left_point_distance)
            
                        if steering < -.5:
                            steering = -.5
                        elif steering > .5:
                            steering = .5

                        angle = potingen_straße(steering, POT_CENTER-1800)

                        if park_coordinate > 1600:
                            experimental_park_stage_1 = True

                        mahmut.adc0 = int(0)
                        mahmut.adc1 = int(angle)
                        mahmut.adc2 = 0
                        mahmut.adc3 = FORWARD
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)
                    
                    elif experimental_park_stage_1 and not experimental_park_stage_2:
                        mahmut.adc0 = int(AUTONOMOUS_SPEED)
                        mahmut.adc1 = int(FULL_RIGHT) # ölümüne sağ
                        mahmut.adc2 = 0
                        mahmut.adc3 = FORWARD
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)

                        if abs(850 - park_coordinate) < 100:
                            experimental_park_stage_2 = True

                    else:
                        print("EXPERiMENTAL PARKiNG LasT StaGE")
                        target_diff = (850 - park_coordinate) / 1700
                        steer = (target_diff + 0.5) * 3600

                        mahmut.adc0 = int(AUTONOMOUS_SPEED)
                        mahmut.adc1 = int(steer) # face your destiny mahmut
                        mahmut.adc2 = 0
                        mahmut.adc3 = FORWARD
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)

                        if distances['on'] < 1.5:
                            while True:
                                mahmut.adc0 = DURMAK
                                mahmut.adc1 = 1800
                                mahmut.adc2 = 11000
                                mahmut.adc3 = FORWARD
                                mahmut.adc4 = True
                                mahmut.adc5 = int(driving_mode)
                                arduino.publish(mahmut)
                                print("DURDUM \m/!")
                                time.sleep(1/20)


                # raw two third mode
                elif 0:
                    if distances['on'] > 7.5:
                        print("TWO THIRD PARK")
                        target_diff = (twothird - park_coordinate) / 1700
                        steer = (target_diff + 0.5) * 3600
                    else:
                        print("MIDDLE PARk")
                        target_diff = (850 - park_coordinate) / 1700
                        steer = (target_diff + 0.5) * 3600

                        if distances['on'] < 1.5:
                            while True:
                                mahmut.adc0 = DURMAK
                                mahmut.adc1 = 1800
                                mahmut.adc2 = 11000
                                mahmut.adc3 = FORWARD
                                mahmut.adc4 = True
                                mahmut.adc5 = int(driving_mode)
                                arduino.publish(mahmut)
                                print("DURDUM \m/!")
                                time.sleep(1/20)

                    mahmut.adc0 = AUTONOMOUS_SPEED
                    mahmut.adc1 = int(steer)
                    mahmut.adc2 = 0
                    mahmut.adc3 = FORWARD
                    mahmut.adc4 = True
                    mahmut.adc5 = int(driving_mode)

                # first left track, düzgün park distance sonrası two_third/mid tracking
                elif 1:
                    #stagei buradan kaldır ya da adını degistir xd
                    if stage1:
                        print(bcolors.FAIL+"ONEMLİ BURAYA GİRİYOR MU 1, sollu"+bcolors.ENDC)
                        right_point_distance = np.average(right_array)
                        left_point_distance = np.average(sol_array)

                        if(right_point_distance > 5):
                            right_point_distance = 5
                        if(left_point_distance > 5):
                            left_point_distance = 5

                        parksagdistance = 1.55
                        steering = fast_pp(parksagdistance, left_point_distance)

                        if steering < -.5:
                            steering = -.5
                        elif steering > .5:
                            steering = .5
                        
                        angle = potingen_straße(steering, POT_CENTER-1800)

                        if park_coordinate > 1500:
                            stage1 = False

                        # doldur
                        mahmut.adc0 = int(37)
                        mahmut.adc1 = int(angle)
                        mahmut.adc2 = 0
                        mahmut.adc3 = FORWARD
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)
                    else:
                        print("stage1 in else kısmı")
                        if park_distance > 6.1 and yavizseko: #yaviz
                            print("TWO THIRD PARK")
                            target_diff = (twothird - park_coordinate) / 1700
                            steer = (target_diff + 0.5) * 3600
                        else:
                            print("MIDDLE PARk")
                            target_diff = (850 - park_coordinate) / 1700
                            steer = (target_diff + 0.5) * 3600
                        
                        mahmut.adc0 = int(37)
                        mahmut.adc1 = int(steer)
                        mahmut.adc2 = 0
                        mahmut.adc3 = FORWARD
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)                        
                        
                        if distances['on'] < 2.5:
                            mahmut.adc0 = DURMAK
                            mahmut.adc1 = 1900
                            mahmut.adc2 = 11000
                            mahmut.adc3 = FORWARD
                            mahmut.adc4 = True
                            mahmut.adc5 = int(driving_mode)
                            print(bcolors.FAIL+"DURMAK \m/!"+bcolors.ENDC)                        
                        
                        elif distances['on'] < 3.1:
                            mahmut.adc0 = int(37)
                            mahmut.adc1 = 2000
                            mahmut.adc2 = 0
                            mahmut.adc3 = FORWARD
                            mahmut.adc4 = True
                            mahmut.adc5 = int(driving_mode)
                            print(bcolors.FAIL+"YAVASLAMAK \m/!"+bcolors.ENDC)




                
                
                # two third, mid, geri 2x, geri x, düz
                elif 0:
                    if stage1:
                        if park_distance > 8:
                            print("TWO THIRD PARK")
                            target_diff = (twothird - park_coordinate) / 1700
                            steer = (target_diff + 0.5) * 3600
                            mahmut.adc0 = 25
                            mahmut.adc1 = int(steer)
                            mahmut.adc2 = 0
                            mahmut.adc3 = FORWARD
                            mahmut.adc4 = True
                            mahmut.adc5 = int(driving_mode)
                        
                            if distances['on'] < 3.5:
                                stage1 = False
                                stage2 = True
                    elif stage2:
                        mahmut.adc0 = 25
                        mahmut.adc1 = FULL_LEFT
                        mahmut.adc2 = 0
                        mahmut.adc3 = REVERSE
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)
                        
                        if distances['on'] < 6.0:
                            stage2 = False
                            stage3 = True
                    elif stage3:
                        mahmut.adc0 = 25
                        mahmut.adc1 = FULL_RIGHT
                        mahmut.adc2 = 0
                        mahmut.adc3 = REVERSE
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)
                        
                        if distances['on'] < 7.5:
                            stage3 = False
                            stage4 = True
                    elif stage4:
                        target_diff = (640 - park_coordinate) / 1280

                        mahmut.adc0 = 31
                        mahmut.adc1 = (target_diff + 0.5) * 3600
                        mahmut.adc2 = 0
                        mahmut.adc3 = FORWARD
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)

                        if distances['on'] < 2.:
                            while True:
                                mahmut.adc0 = DURMAK
                                mahmut.adc1 = 1800
                                mahmut.adc2 = 11000
                                mahmut.adc3 = FORWARD
                                mahmut.adc4 = True
                                mahmut.adc5 = int(driving_mode)
                                arduino.publish(mahmut)
                                print("DURDUM \m/!")
                                time.sleep(1/20)


            # anciecnt park koordinatı bilinen kod
            elif 0:
                pass
                # zed pose beklenen dişinda
                #@ZED değiştirilecek !!!!
                parking_object.__current_pos = [zed_pose[0], zed_pose[1]] # anlık araç konumu
                parking_object.__eval__()
                
                # left tracking
                if not parking_object.__destiny:
                    right_point_distance = np.average(right_array)
                    left_point_distance = np.average(sol_array)

                    if(right_point_distance > 5):
                        right_point_distance = 5
                    if(left_point_distance > 5):
                        left_point_distance = 5
                    
                    steering = fast_pp(SAG_FIXED, left_point_distance)
        
                    if steering < -.5:
                        steering = -.5
                    elif steering > .5:
                        steering = .5

                    angle = potingen_straße(steering, POT_CENTER-1800)

                    mahmut.adc0 = int(0)
                    mahmut.adc1 = int(angle)
                    mahmut.adc2 = 0
                    mahmut.adc3 = FORWARD
                    mahmut.adc4 = True
                    mahmut.adc5 = int(driving_mode)
                
                
                # face your destiny dear mahmut
                else:
                    # stopping condition
                    if distances['on'] < 1.5:
                        mahmut.adc0 = int(0)
                        mahmut.adc1 = int(1800)
                        mahmut.adc2 = 11000
                        mahmut.adc3 = FORWARD
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)
                    
                    elif distances['on'] < LABEL_OFFSET and parking_sign_current_distance < LABEL_OFFSET:
                        print("Middle Takipppp")
                        target_diff = (640 - park_coordinate) / 1280
                        steer = (target_diff + 0.5) * 3600

                        mahmut.adc0 = int(AUTONOMOUS_SPEED)
                        mahmut.adc1 = int(steer)
                        mahmut.adc2 = 0
                        mahmut.adc3 = FORWARD
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)
                    # maneuver mode
                    else:
                        mahmut.adc0 = int(AUTONOMOUS_SPEED)
                        mahmut.adc1 = int(FULL_RIGHT) # ölümüne sağ
                        mahmut.adc2 = 0
                        mahmut.adc3 = FORWARD
                        mahmut.adc4 = True
                        mahmut.adc5 = int(driving_mode)

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
            stage1 = True
            pid_controller.pidError = 0
            hazreticounter = 0
            is_parking_mode = False
            steering_angle = (l_left_right+1)*1800  # 0 3600
            experimental_park_stage_1 = False
            experimental_park_stage_2 = False

            #if not CRUISE_CONTROL and driving_mode == CURRENT_MODE:
            if driving_mode == CURRENT_MODE:
                speed = mapper(right_trigger, 1, -1, 0, 1000)
            #elif not CRUISE_CONTROL and driving_mode == RPM_MODE:
            elif driving_mode == RPM_MODE:
                speed = mapper(right_trigger, 1, -1, 0, MAX_RPM_MODE_SPEED)            
            
            regen = mapper(left_trigger, 1, -1, 0, 1000)

            mahmut.adc0 = int(speed)           # speed (0, 1000)
            mahmut.adc1 = int(steering_angle)  # steering angle (0, 3600)
            mahmut.adc2 = int(regen + brake_value)           # regen (0, 1000)
            mahmut.adc3 = int(GEAR)            # 
            mahmut.adc4 = int(AUTONOMOUS)      # autonomous
            mahmut.adc5 = int(driving_mode)    # mode

            if distances['left'] - distances['right'] > 0.1:
                d = "left"
            elif distances['left'] - distances['right'] < -0.1:
                d = "right"            
            else:
                d = "Mid"

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
        print(bcolors.FAIL + "404 FATAL ERROR" + bcolors.ENDC)

    print("sag uzaklık:", np.average(right_array))
    print("sol uzaklık:", np.average(sol_array))

    control_mahmut(mahmut)
    arduino.publish(mahmut)



def F1_2020(russell):
    """     AXIS:
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
            10: BUTTON STICK RIGHT """

    global YAVIZ
    global SEKO
    global TERMINATOR
    global speed
    global brake
    global brake_value
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

        # trigger safety
        left_trigger = 1
        right_trigger = 1

        if YAVIZ and SEKO:
            left_trigger = russell.axes[2]
            right_trigger = russell.axes[5]
            
            # A B X Y #
            if BUTTON_Y:
                AUTONOMOUS ^= True
            if BUTTON_A:
                driving_mode ^= True
            if BUTTON_B:
                brake ^= True
                if brake:
                    brake_value = 0
                else:
                    brake_value = 10000
                #collecting_data ^= True
            if BUTTON_X and right_trigger == 1:
                GEAR = next(gear_generator)
            if BUTTON_BACK:
                CRUISE_CONTROL ^= True
                if CRUISE_CONTROL == False: speed = 0
            if BUTTON_LB:
                if AUTONOMOUS:
                    if AUTONOMOUS_SPEED != 0:
                        AUTONOMOUS_SPEED -= 25
                elif CRUISE_CONTROL:
                    if speed != 0:
                        speed -= 25
                else:
                    ...
            
            if BUTTON_RB:
                if AUTONOMOUS:
                    if AUTONOMOUS_SPEED < 100:
                        AUTONOMOUS_SPEED += 25
                elif CRUISE_CONTROL:
                    if speed < 200:
                        speed += 25
                else:
                    ...
            
            if BUTTON_STICK_RIGHT:
                is_curve_parking ^= True


def yolo_callback(data):
    global AUTONOMOUS_SPEED
    global AUTONOMOUS_SPEED_RECOVERY
    global CRITICAL_PARKING_DISTANCE
    global CALCULATE_PARKING_SIGN_DISTANCE
    global TWO_THIRD_PARKING
    global REVERSE_PARKING
    global speed
    global brake
    global brake_value
    global is_parking_mode
    global kararVerici
    global left_tracking
    global right_tracking
    global mid_tracking
    global first_stop_counter
    global first_stop
    global zed_x
    global zed_y
    global zed_z
    global park_distance
    global durak_dik_start
    global is_curve_created
    global closest_point
    global recently_stopped
    global orhandaldal
    global roswtf
    global parking_object
    global locked_on_target
    global parking_sign_current_distance
    global recently_stopped_kirmizi
    global left_point_distance
    global right_point_distance
    global kirmizida_dur_lan
    global karar_verici_yakın_zamanda_calisti
    global ahaburasıdaboşmuşıheahıeah
    global olceriz_sıkıntı_yog
    global sol_array
    global right_array
    global hazreticounter
    global park_left_not_started
    global park_coordinate
    global stage1
    global stage2
    global stage3
    global stage4

    global durak_hanzonun_yardımcısı
    global mahmutapozisyonçeşitliliği
    global yesil_gorundu
    global kirmizi_distance

    global girildim_bool
    global girildim_counter

    sola_donulmez_goruldu = False
    saga_donulmez_goruldu = False
    girilmez_goruldu = False

    global karsiya_geciyorum

    if karsiya_geciyorum:
        return

    if data.data != "":
        datas = data.data.split(';')
        datas.remove('')
        
        kirmizi_hattori = False
        durak_hanzo = False

        park_seen = False
        park_yapilmaz_seen = False

        for tabela in datas:
            label, _, _, _, distance = tabela.split(',')
            distance = float(distance)

            if label == "kirmizi isik" and float(distance) < 8 and float(distance) > 4. and AUTONOMOUS:
                kirmizi_distance = distance
                kirmizi_hattori = True
                yesil_gorundu = False
            elif label == "yesil isik":
                yesil_gorundu = True
                kirmizi_hattori = False
                kirmizida_dur_lan = False
            elif label == "Durak" and float(distance) < 5.31 and float(distance) > 2 and not recently_stopped and AUTONOMOUS:
                durak_hanzo = True
            elif label == "Park Yasak" and distance < 9.3:
                hazreticounter += 1



        if kirmizi_hattori:
            kirmizida_dur_lan = True
            return


        if durak_hanzo:
            print("durak hanzo")
            roswtf = True
            recently_stopped = True
            durak_dik_start = [zed_x, zed_z]
            return

        sola_donulmez_goruldu = False
        saga_donulmez_goruldu = False
        girilmez_goruldu = False

        r_u_sure = False

        if hazreticounter > 10:
            #print(bcolors.WARNING + "AAAAAAA" * 1000 + bcolors.ENDC)
            mid_tracking = False
            left_tracking = False
            right_tracking = False
            roswtf = False
            is_parking_mode = True


        for tabela in datas:
            label, x, y, z, distance = tabela.split(',')
            
            x = float(x)
            y = float(y)

            distance = float(distance)
            
            ##################################################################################################
            ##################################################################################################
            ###################################################################################################
            ###################################################################################################
            ##################################################################################################            
            if label == "yesil isik":
                yesil_gorundu = True
            elif label == "Park Yeri":
                pass
                #@ZED değiştirilecek !!!!
                #parking_object.__target = [x, y] # hedef tabelanın koordinatlarını belirle!
                #locked_on_target = True

            #elif label in ("Park Yeri", "Park Yapilmaz") and distance < CRITICAL_PARKING_DISTANCE:
            #    is_parking_mode = True
            #    break
            ###################################################################################################
            ###################################################################################################
            ###################################################################################################
            ###################################################################################################
            ###################################################################################################
            elif label == "sola donulmez" and distance < 6.5 and distance > 3.5:
                sola_donulmez_goruldu = True
            elif label == "saga donulmez" and distance < 6.5 and distance > 3.5:
                saga_donulmez_goruldu = True
            elif label == "Girilmez" and distance < 15 and distance > 3.5:
                print("GİRİLDİM")
                if girildim_bool:
                    if girildim_counter == 0:
                        left_tracking = True
                        right_tracking = False
                        mid_tracking = False
                        girildim_bool = False
                    elif girildim_counter == 1:
                        left_tracking = False
                        right_tracking = True
                        mid_tracking = False
                        girildim_bool = False
                    elif girildim_counter == 2:
                        left_tracking = True
                        right_tracking = False
                        mid_tracking = False
                        girildim_bool = False
                    
                    

                #girilmez_goruldu = True
            ####################################################################################
            elif label == "ileriden sola mecburi yon" and distance < 6. and distance > 3.5:
                left_tracking = True
                right_tracking = False
                mid_tracking = False
                r_u_sure = True
            elif label == "ileriden saga mecburi yon" and distance < 6. and distance > 3.5:
                left_tracking = False
                right_tracking = True
                mid_tracking = False
                r_u_sure = True
            ####################################################################################
            elif label == "ileri Ve saga mecburi yon" and distance < 6. and distance > 3.5:
                left_tracking = False
                right_tracking = True
                mid_tracking = False
                r_u_sure = True
            elif label == "ileri Ve sola mecburi yon" and distance < 6. and distance > 3.5:
                right_tracking = False
                left_tracking = True
                mid_tracking = False
                r_u_sure = True
            ####################################################################################
            else:
                pass
        
        if not r_u_sure:
            if sola_donulmez_goruldu and saga_donulmez_goruldu:               
                left_tracking = False
                mid_tracking = True
                right_tracking = False
                karsiya_geciyorum = True

            elif sola_donulmez_goruldu:
                left_tracking = False
                mid_tracking = False
                right_tracking = True
            elif saga_donulmez_goruldu:
                left_tracking = True
                mid_tracking = False
                right_tracking = False

        # OLD BUT GOLD  ahaburasıdaboşmuşıheahıeahahaburasıdaboşmuşıheahıeahahaburasıdaboşmuşıheahıeah   
        """
        if r_u_sure:
            pass
        elif not a and (girilmez_goruldu or saga_donulmez_goruldu or sola_donulmez_goruldu):
            karar_verici_yakın_zamanda_calisti = True
            karar_verici_start_time = time.time()
            if girilmez_goruldu and sola_donulmez_goruldu and not saga_donulmez_goruldu:
                mid_tracking = False
                left_tracking = False
                right_tracking = True
            elif girilmez_goruldu and saga_donulmez_goruldu and not sola_donulmez_goruldu:
                mid_tracking = False
                left_tracking = True
                right_tracking = False
            elif saga_donulmez_goruldu and sola_donulmez_goruldu and not girilmez_goruldu:
                pass
            elif girilmez_goruldu and not sola_donulmez_goruldu and not saga_donulmez_goruldu:
                if left_point_distance < right_point_distance:
                    right_tracking = True
                    mid_tracking = False
                    left_tracking = False
                elif right_point_distance < left_point_distance:
                    left_tracking = True
                    mid_tracking = False
                    right_tracking = False
            elif sola_donulmez_goruldu and not saga_donulmez_goruldu and not girilmez_goruldu:
                mid_tracking = False
                left_tracking = False
                right_tracking = True
            elif saga_donulmez_goruldu and not sola_donulmez_goruldu and not girilmez_goruldu:
                mid_tracking = False
                left_tracking = True
                right_tracking = False
            """
    
                
def park_coordinate_callback(park_data):
    global park_coordinate
    if park_data.data == '0':
        pass
    else:
        park_coordinate = float(park_data.data)

def park_distance_callback(data):
    global park_distance
    if data.data == '0':
        pass
    else:
        park_distance = float(data.data)

# Point(x, y, z) data
def zed_pose(data):
    global zed_x
    global zed_y
    global zed_z

    zed_x = data.x
    zed_y = data.y
    zed_z = data.z
    
"""
# Keyboard #
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
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
"""
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



if __name__ == "__main__":
    # this node #
    if len(sys.argv) == 2:
        yavizseko = int(sys.argv[1])
    else:
        yavizseko=1            

    rospy.init_node('mahmut',anonymous=True)

    # saving data
    file = open("çöp.csv", "a+")
    writer = csv.writer(file)
    writer.writerow(["SOL", "SAG", "DIREKSIYON"])

    # steering angle #
    pid_controller = Steering_Algorithms.PID(0.8, 0.0075, 0.225)
    pürşit = Steering_Algorithms.Pure_Pursuit_Controller(5, 5, 1)

    # just in time compiler #
    print(potingen_straße(31, 31))
    print(ackermann(31, 31, 31))
    print(fast_pp(31, 31))
    print(fast_pid(31,31,31,31,31,31))
    # safety #
    #AUTONOMOUS_SPEED -= (AUTONOMOUS_SPEED % 25)
    
    # subscribers
    rospy.Subscriber('/scan', LaserScan, lidar_data, queue_size=10)
    rospy.Subscriber('/joy', Joy, F1_2020, queue_size=10)
    rospy.Subscriber('/pot_topic', Adc, haydi_gel_icelim, queue_size=10)
    rospy.Subscriber('yolo_park', String, park_coordinate_callback, queue_size=10)
    rospy.Subscriber('/yolo_park_distance', String, park_distance_callback, queue_size=10)
    rospy.Subscriber('/zed_detections', String, yolo_callback, queue_size=10)
    rospy.Subscriber('zed_pose', Point, zed_pose, queue_size = 10)
    # publishers
    arduino = rospy.Publisher("/seko", Adc, queue_size=10, latch=True)
    lcd = rospy.Publisher("/screen", Adc, queue_size=10, latch=True)

    # rosmsg #
    f710 = Joy()
    mahmut = Adc()
    park = serhatos()

    # Parking #
    parking_object = park_seko(ACKERMAN_RADIUS, LABEL_OFFSET, ERROR_ACCEPTANCE)

    # driving with keyboard    
    hiz = 0
    direk = 1800
    #keyboard_listener = Listener(on_press=on_press, on_release=on_release)
    #keyboard_listener.start()
    
    while not rospy.is_shutdown():
        rospy.spin()

    file.close()
