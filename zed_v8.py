#!/usr/bin/env python3

import os
import sys
import time
import logging
import random
from random import randint
import math
import statistics
import getopt
from ctypes import *
import numpy as np
import cv2
import pyzed.sl as sl
import joblib
import matplotlib.pyplot as plt
from skimage.feature import hog
from PIL import Image as PImage
import rospy
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String
#from keras.models import load_model
from cv_bridge import CvBridge, CvBridgeError




# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                log.info("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # log.info(os.environ.keys())
            # log.warning("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            log.warning("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            log.warning("Environment variables indicated a CPU run, but we didn't find `" +
                        winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("/home/otonom/otonom_ws/src/zed_package/src/zed-yolo/libdarknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            name_tag = meta.names[i]
        else:
            name_tag = altNames[i]
        res.append((name_tag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.8, hier_thresh=.8, nms=.45, debug=False):
    """
    Performs the detection
    """
    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(
        net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    if debug:
        log.debug("about to range")
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    name_tag = meta.names[i]
                else:
                    name_tag = altNames[i]
                res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res


netMain = None
metaMain = None
altNames = None


def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.

    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 10

    x_vect = []
    y_vect = []
    z_vect = []
    if(int(bounds[0]) > 1700 or int(bounds[0]) < 220 or int(bounds[1]) > 960 or int(bounds[1]) < 121):
        x_median = -1
        y_median = -1
        z_median = -1
    else:
        for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
            for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
                z = depth[i, j, 2]
                if not np.isnan(z) and not np.isinf(z):
                    x_vect.append(depth[i, j, 0])
                    y_vect.append(depth[i, j, 1])
                    z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median



cam = sl.Camera()
runtime = sl.RuntimeParameters()
mat = sl.Mat()
point_cloud_mat = sl.Mat()
thresh = 0.25
color_array = 0




left_right_model = joblib.load('/home/otonom/otonom_ws/src/zed_package/src/zed-yolo/zed_python_sample/yolo_data/hattori.npy')
park_durak_model = joblib.load('/home/otonom/otonom_ws/src/zed_package/src/zed-yolo/zed_python_sample/yolo_data/durak_park.npy')
#traffic_light_model = load_model('/home/otonom/otonom_ws/src/zed_package/src/zed-yolo/zed_python_sample/yolo_data/traffic_light_model.h5')
must_lr_model = joblib.load('/home/otonom/otonom_ws/src/zed_package/src/zed-yolo/zed_python_sample/yolo_data/must_r_l.npy')


orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3
runtime = 0
zed_pose = 0

zed_sensors = 0

def left_right_judgement(img,x_min,x_max,y_min,y_max):
    #tespit edilen levhayi ikiye bolduk ama ise yaramadi
    #cropped_img_left = img[y_min:y_max, x_min:int(x_min+(box_width)/2)]
    #cropped_img_right = img[y_min:y_max, x_min+int(box_width/2):x_max]
    cropped_img = img[y_min:y_max, x_min:x_max]
    #cropped_img = img

    if not all(cropped_img.shape):
        return 0

    img = PImage.fromarray(cropped_img)
    img = img.resize((128,128))
    gray= img.convert('L')
    # Now we calculate the HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 

    fd = fd.reshape(1,8100)
    sonuc = left_right_model.predict(fd)
    return sonuc


def park_durak_judgement(img,x_min,x_max,y_min,y_max):
    cropped_img = img[y_min:y_max, x_min:x_max]
    #cropped_img = img

    if not all(cropped_img.shape):
        return 0

    img = PImage.fromarray(cropped_img)
    img = img.resize((128,128))
    gray= img.convert('L')
    # Now we calculate the HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 

    fd = fd.reshape(1,8100)
    sonuc = park_durak_model.predict(fd)
    return sonuc

"""def traffic_light_judgement(img, x_min,x_max,y_min,y_max):
    if not all(img.shape):
        return 0
    img = img[y_min:y_max, x_min:x_max]
    desired_dim=(32,32)
    img_resized = cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
    img_ = np.expand_dims(np.array(img_resized), axis=0)

    predicted_state =np.argmax(traffic_light_model.predict(img_), axis=-1)

    return predicted_state"""



def must_left_right_judgement(img, x_min,x_max,y_min,y_max):
    cropped_img = img[y_min:y_max, x_min:x_max]
    #cropped_img = img

    if not all(cropped_img.shape):
        return 0

    img = PImage.fromarray(cropped_img)
    img = img.resize((128,128))
    gray= img.convert('L')
    # Now we calculate the HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 

    fd = fd.reshape(1,8100)
    sonuc = must_lr_model.predict(fd)
    return sonuc



def main():
    global color_array
    global cam
    global runtime
    global mat
    global point_cloud_mat
    global thresh
    global runtime
    global zed_pose
    global zed_sensors


    darknet_path="/home/otonom/otonom_ws/src/zed_package/src/zed-yolo/libdarknet/"
    config_path = "/home/otonom/otonom_ws/src/zed_package/src/zed-yolo/zed_python_sample/yolo_data/ismet_yolov3_v2.cfg"
    weight_path = "/home/otonom/otonom_ws/src/zed_package/src/zed-yolo/zed_python_sample/yolo_data/ismet_yolov3_v2_best_700k.weights"
    meta_path = "/home/otonom/otonom_ws/src/zed_package/src/zed-yolo/zed_python_sample/yolo_data/coco.data"
    svo_path = None
    zed_id = 0

    help_str = 'darknet_zed.py -c <config> -w <weight> -m <meta> -t <threshold> -s <svo_file> -z <zed_id>'
    try:
        opts, args = getopt.getopt(
            [], "hc:w:m:t:s:z:", ["config=", "weight=", "meta=", "threshold=", "svo_file=", "zed_id="])
    except getopt.GetoptError:
        log.exception(help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            log.info(help_str)
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg
        elif opt in ("-w", "--weight"):
            weight_path = arg
        elif opt in ("-m", "--meta"):
            meta_path = arg
        elif opt in ("-t", "--threshold"):
            thresh = float(arg)
        elif opt in ("-s", "--svo_file"):
            svo_path = arg
        elif opt in ("-z", "--zed_id"):
            zed_id = int(arg)

    input_type = sl.InputType()
    if svo_path is not None:
        log.info("SVO file : " + svo_path)
        input_type.set_from_svo_file(svo_path)
    else:
        # Launch camera by id
        input_type.set_from_camera_id(zed_id)

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 15  # Use HD720 video mode (default fps: 60)
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.coordinate_units = sl.UNIT.METER  # Set units in meters
    init.depth_mode = sl.DEPTH_MODE.ULTRA

    
    #cam.enable_streaming()
    if not cam.is_opened():
        log.info("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()


    py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    err = cam.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Track the camera position during 1000 frames

    zed_pose = sl.Pose()

    zed_sensors = sl.SensorsData()

    runtime = sl.RuntimeParameters()
    # Use STANDARD sensing mode
    runtime.sensing_mode = sl.SENSING_MODE.FILL
    mat = sl.Mat()
    point_cloud_mat = sl.Mat()

    # Import the global variables. This lets us instance Darknet once,
    # then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(config_path)+"`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weight_path)+"`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(meta_path)+"`")
    if netMain is None:
        netMain = load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(meta_path.encode("ascii"))
    if altNames is None:
        # In thon 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(meta_path) as meta_fh:
                meta_contents = meta_fh.read()
                import re
                match = re.search("names *= *(.*)$", meta_contents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as names_fh:
                            names_list = names_fh.read().strip().split("\n")
                            altNames = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass


    log.info("Running...")

    key = ''

x_orta = 0

y_orta = 0

def SEBASTIAN_VETTEL():
    global x_orta
    global y_orta

    states = ['red', 'yellow', 'green', 'off']

    label = '' 
    start_time = time.time() # start time of the loop
    err = cam.grab(runtime)
    distance = 0
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        image = mat.get_data()
        raw_image = image.copy()
        image = adjust_gamma(image, 2)
        cam.retrieve_measure(
            point_cloud_mat, sl.MEASURE.XYZRGBA)
        depth = point_cloud_mat.get_data()

        # Do the detection
        detections = detect(netMain, metaMain, image, thresh)

        log.info(chr(27) + "[2J"+"**** " + str(len(detections)) + " Results ****")
        detected_objects = ""
        for detection in detections:
                label = detection[0]
                label = str(label)
                confidence = detection[1]

                bounds = detection[2]
                y_extent = int(bounds[3])
                x_extent = int(bounds[2])
                # Coordinates are around the center
                x_coord = int(bounds[0] - bounds[2]/2)
                y_coord = int(bounds[1] - bounds[3]/2)
                #boundingBox = [[x_coord, y_coord], [x_coord, y_coord + y_extent], [x_coord + x_extent, y_coord + y_extent], [x_coord + x_extent, y_coord]]
                thickness = 1

                x, y, z = get_object_depth(depth, bounds)


                if((label=='sola donulmez') or (label=='saga donulmez')):
                    #cv2.imshow("result",image)
                    result = left_right_judgement(image,x_coord ,x_coord + x_extent,y_coord,y_coord + y_extent)
                    if(result):
                        label = "sola donulmez"
                    else:
                        label = "saga donulmez"

                if((label == 'Durak') or (label == "Park Yeri")):
                    result = park_durak_judgement(image,x_coord ,x_coord + x_extent,y_coord,y_coord + y_extent)
                    if(result):
                        label = "Park Yeri"
                    else:
                        label = "Durak"
                
                if((label == 'ileriden sola mecburi yon') or (label == "ileriden saga mecburi yon") or (label == "Sola Mecburi Yon") or (label == "Saga Mecburi Yon")):
                    result = must_left_right_judgement(image,x_coord ,x_coord + x_extent,y_coord,y_coord + y_extent)
                    if(result):
                        label = "ileriden sola mecburi yon"
                    else:
                        label = "ileriden saga mecburi yon"

                """if((label == 'yesil isik') or (label == "kirmizi isik") or (label == "sari isik") or (label == "Trafik Lambasi")):
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

                    result = traffic_light_judgement(image,x_coord ,x_coord + x_extent,y_coord,y_coord + y_extent)
                    for idx in result:
                        result = (states[idx])
                        if(result == "green"):
                            label = "yesil isik"
                        elif(result == "red"):
                            label == "kirmizi isik" """
                        
                if(label == "Park Yeri"):
                    x_orta = x_coord + x_extent / 2
                    y_orta = y_coord + y_extent / 2

                
                pstring = label+": "+ str(np.rint(100 * confidence))+"%"
                log.info(pstring)
                distance = math.sqrt(x * x + y * y + z * z)
                sign_coord.x = x
                sign_coord.y = y
                sign_coord.z = z
                distance = "{:.2f}".format(distance)
                cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                              (x_coord + x_extent + thickness, y_coord + (18 + thickness*4)),
                              (37,66,0), -1)
                cv2.putText(image, label + " " +  (str(distance) + " m"),
                            (x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                              (x_coord + x_extent + thickness, y_coord + y_extent + thickness),
                              (37,66,0), int(thickness*2))
                if(x == -1 and y == -1 and z == -1):
                    pass
                elif (distance == '1.73'):
                    pass
                else:
                    detected_objects += label + "," + str(x)[0:4]+ ","+ str(y)[0:4] + "," + str(z)[0:4] + ","+ str(distance) + ";"


        cv2.imshow("ZED", image)
        key = cv2.waitKey(5)
        log.info("FPS: {}".format(1.0 / (time.time() - start_time)))
    else:
        key = cv2.waitKey(5)
    

    label_and_distance = label + ' ' +str(distance)
    return raw_image, label_and_distance, sign_coord, detected_objects


s_time = time.time()
zed_odom_speed = None
previous_pos = np.array([0, 0, 0], dtype=np.float32)

def SAINZ():
    global s_time
    global zed_odom_speed
    global previous_pos
    global runtime
    global zed_pose
    global cam
    global zed_sensors

    if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Get the pose of the left eye of the camera with reference to the world frame
        cam.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
        cam.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
        zed_imu = zed_sensors.get_imu_data()

        # Display the translation and timestamp
        py_translation = sl.Translation()
        tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
        ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
        tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
        #print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))
        pose_msg.x = odometry_msg.pose.pose.position.x = tx
        pose_msg.y = odometry_msg.pose.pose.position.y = ty
        pose_msg.z = odometry_msg.pose.pose.position.z = tz

        initial_time = time.time()
        diff = s_time - initial_time   

        odometry_msg.twist.twist.linear.z = round((previous_pos[0]-pose_msg.z) / diff, 3)
        odometry_msg.twist.twist.linear.x = round((previous_pos[1]-pose_msg.x) / diff, 3)
        odometry_msg.twist.twist.linear.y = round((previous_pos[2]-pose_msg.y) / diff, 3)
        previous_pos[0] = pose_msg.z
        previous_pos[1] = pose_msg.x
        previous_pos[2] = pose_msg.y

        """         zed_odom_speed = math.sqrt(pow(previous_pos[0]-pose_msg.x , 2)+
                                   pow(previous_pos[1]-pose_msg.y , 2))
    
        zed_odom_speed /= diff

        previous_pos[0] = momentary_x
        previous_pos[1] = momentary_y """
        s_time = initial_time 

        # Display the orientation quaternion
        py_orientation = sl.Orientation()
        odometry_msg.pose.pose.orientation.x = ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
        odometry_msg.pose.pose.orientation.y = oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
        odometry_msg.pose.pose.orientation.z = oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
        odometry_msg.pose.pose.orientation.w = ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
        #print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))
        
        #Display the IMU acceleratoin
        acceleration = [0,0,0]
        zed_imu.get_linear_acceleration(acceleration)
        ax = round(acceleration[0], 3)
        ay = round(acceleration[1], 3)
        az = round(acceleration[2], 3)
        #print("IMU Acceleration: Ax: {0}, Ay: {1}, Az {2}\n".format(ax, ay, az))
        
        #Display the IMU angular velocity
        a_velocity = [0,0,0]
        zed_imu.get_angular_velocity(a_velocity)
        odometry_msg.twist.twist.angular.x = vx = round(a_velocity[0], 3)
        odometry_msg.twist.twist.angular.y = vy = round(a_velocity[1], 3)
        odometry_msg.twist.twist.angular.z = vz = round(a_velocity[2], 3)
        #print("IMU Angular Velocity: Vx: {0}, Vy: {1}, Vz {2}\n".format(vx, vy, vz))

        # Display the IMU orientation quaternion
        zed_imu_pose = sl.Transform()
        ox = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[0], 3)
        oy = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[1], 3)
        oz = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[2], 3)
        ow = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[3], 3)
        #print("IMU Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))
        return pose_msg, odometry_msg


def mapper(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def test_uzaklik(data):
    global x_orta
    global y_orta

    array_first = data.ranges[0:240]
    array_second = data.ranges[1199:1439]

    distance_120 = np.concatenate((array_first[::-1],array_second[::-1]))

    mapped_x = int(mapper(x_orta, 0,1280,0,480))

    sign_lidar_distance = distance_120[int(mapped_x/4)]

    print("Lidar distance of sign: ", sign_lidar_distance,"x_orta",x_orta)



main()

rospy.init_node("zed_node",anonymous=True)
pub = rospy.Publisher('zed_yolo_raw_image',Image,queue_size=10)
pub_ = rospy.Publisher('zed_yolo_raw_distance',String,queue_size=10)
pose_pub = rospy.Publisher('zed_yolo_pose',Point,queue_size=10)
sign_pub = rospy.Publisher('zed_yolo_sign_coord', Point, queue_size=10)
odometry_pub = rospy.Publisher('odom', Odometry, queue_size=10)
detections_pub = rospy.Publisher('zed_detections', String, queue_size=10)


image_msgs = Image()

pose_msg = Point()

sign_coord = Point()

odometry_msg = Odometry()

odometry_msg.child_frame_id = "base_link"
  
bridge = CvBridge()

while not rospy.is_shutdown():
    
    image, label_and_distance, sign_coord, detected_objects = SEBASTIAN_VETTEL()
    pose_msg, odometry_msg = SAINZ()

    image_message = bridge.cv2_to_imgmsg(image, "passthrough")

    
    rospy.Subscriber("/scan", LaserScan, test_uzaklik)

    pub.publish(image_message)
    pub_.publish(str(label_and_distance))
    pose_pub.publish(pose_msg)
    sign_pub.publish(sign_coord)
    detections_pub.publish(detected_objects)
    odometry_pub.publish(odometry_msg)
    
