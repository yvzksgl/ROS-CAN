/*
    @year:        2020/2021
    @author:      Sekomer
    @touch:       aksoz19@itu.edu.tr
*/


/* Essential Header Files */
#include    <ros.h>
#include    <CAN.h>
#include    <std_msgs/Float32.h>
#include    <geometry_msgs/Twist.h>
#include    <geometry_msgs/Vector3.h>


/* Preprocessing */
#define     PI                  3.1415
#define     RADIUS              0.28
#define     BUFFER_SIZE         32
#define     GEAR_SCALE          7
#define     MAX_STEERING_ANGLE  35
#define     MAX_RPM             350
#define     MAX_REVERSE_RPM    -100


/* Function Declerations */
int32_t buffer_avg(int32_t *buffer, int32_t size);
float radian2degree(float input);
int32_t kmh2rpm (float vel);


/**
*   DATA TYPES FOR CAN COMMUNICATION
*/
typedef union { 
    float   data; 
    uint8_t data_u8[4];
} CAN_Float;


typedef union { 
    uint16_t data_u16[2];
    uint8_t  data_u8[4]; 
} STEER;


/* CAN variable decleartions */
unsigned int motor_collector[8];
unsigned int steer_collector[8];
unsigned int index = 0;

CAN_Float motor_odometry;
CAN_Float steer_odometry;
CAN_Float current;

/* speed, steering and condition info */
CAN_Float rpm;
STEER     steering_obj;
float     regen;
int32_t   current_position = 0;


/* Encoder Variables */
int32_t pot_signal_raw;
int32_t encoder_degree;
int32_t desired_pos;


/* Constant Steering Motor Speeds */
int32_t  const   max_steer_speed = 14;  // change STM side 
int32_t  const   min_steer_speed  = 4;
int32_t  const   high_steer_speed = 10;  // delete


volatile int32_t speed;                   // speed variable CAN
int32_t change_value = 0;                 // momentary change in steering angle

/* Encoder Buffer Variables */
int32_t         buffer_index = 0;
int32_t         buffer[BUFFER_SIZE];
int32_t         buffer_average = 0;
const int32_t   EncoderPin = A0;
/**/

/* Debug and Log Topic */
geometry_msgs::Vector3 pot_data;


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


/*
*     ROS CALLBACK  
*/
void RosCallback(const geometry_msgs::Vector3 &mahmut){
    /* Parsing data from callback data */
    regen = mahmut.z;

    /******************** Speed Logic ******************/ 
    if (mahmut.x >= 0)
        rpm.data = map((long) mahmut.x, 0, 1000, 0, MAX_RPM);    
    else if (mahmut.x < 0)
        rpm.data = map((long) mahmut.x, 0, -1000, 0, MAX_REVERSE_RPM);
    
    if (rpm.data)
        current.data = 0.9;
    else
        current.data = 0.0;

    regen = map((long) mahmut.z, 0, 1000, 0, 100);
    if (regen > 1) {
        current.data = regen / 100;
        rpm.data = 0;
    }
        

    
    /******************** Steering Logic ******************/ 
    /*  
        to send data from terminal or to read data from ROS
        desired_pos = mahmut.y

        to read data from pot add a potansiometer to A1 pin
        desired_pos = direksiyon_pot (A1)
    */
    desired_pos = mahmut.y;
    change_value = desired_pos - current_position; // get change value

    /* Speed Control */
    /*
    if (abs(change_value) < 75)
        speed = low_steer_speed;
    else if (change_value < 350 && change_value > -350)
        speed = high_steer_speed;
    else
        speed = max_steer_speed;
    */
    
    speed = map(abs(change_value), 0, 1800, min_steer_speed, max_steer_speed);
    
    /*
        @steering_obj.data_u16[0]  => steering speed
        @steering_obj.data_u8[2]   => steering 
    */
    if (change_value > 0) {
        steering_obj.data_u16[0] = speed;
        steering_obj.data_u8[2]  = 1;      // steer direction
    }
    else if (change_value < 0) {
        steering_obj.data_u16[0] = speed;
        steering_obj.data_u8[2]  = 0;      // steer direction
    }
    else 
        steering_obj.data_u16[0]  = 0x00;

    /*
     * Driving Motor Packet
    */
    /* rpm */
    CAN.beginPacket(0x501);
    CAN.write(rpm.data_u8[0]);
    CAN.write(rpm.data_u8[1]);
    CAN.write(rpm.data_u8[2]);
    CAN.write(rpm.data_u8[3]);
    /* current */
    CAN.write(current.data_u8[0]);
    CAN.write(current.data_u8[1]);
    CAN.write(current.data_u8[2]);
    CAN.write(current.data_u8[3]);
    CAN.endPacket();

    /* 
     *  Steering Motor Packet
    */
    CAN.beginPacket(0x700);
    CAN.write(steering_obj.data_u8[0]);
    CAN.write(steering_obj.data_u8[1]);
    CAN.write(steering_obj.data_u8[2]);
    CAN.endPacket();
}   




/* 
 *  IT DOESN'T WORK WITH A QUEUE VALUE
 *  @future_debug
*/

/******************** Creating ROS Node ******************/ 
ros::NodeHandle nh;
ros::Subscriber<geometry_msgs::Vector3> sub("/seko", &RosCallback);
ros::Publisher pub("pot_topic", &pot_data);



/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


/*
*    Function to convert kmh to rpm
*/
int32_t kmh2rpm (float vel)
{
    float RPM = 0;
    RPM = 2.65 * vel / RADIUS;

    return RPM;
}

/*
*   Function for calculating buffer average
*/
int32_t buffer_avg(int32_t *buffer, int32_t size)
{
  int32_t sum = 0;
  for (size_t i = 0; i < size; ++i)
    sum += buffer[i];
    
  return sum / size;
}

/*
*   Function for converting radians to degree
*/
float radian2degree(float input)
{
    float degree = input * 180 / PI;
    return degree;
}



/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//                            SETUP AND LOOP                                   //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////



/*
*   Arduino Setup
*
*   rosrun rosserial_arduino serial_node.py _port:=/dev/ttyXXXn _baud:=115200
*/
void setup(){  
    pinMode(A0, INPUT);
    memset(buffer, 0, BUFFER_SIZE * sizeof(int));
    Serial.begin(57600);

    /****************** ROS Initialization ******************/ 
    nh.getHardware()->setBaud(57600);
    nh.initNode();
    nh.subscribe(sub);
    nh.advertise(pub);
    delay(100);

CAN_INIT:
    if(1 == CAN.begin(1E6)) {
      ;/* wait */;
    } 
    else
        goto CAN_INIT;
}


/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


/*
*   Arduino Loop
*
*   rosrun rosserial_arduino serial_node.py _port:=/dev/ttyXXXn _baud:=57600
*/
void loop(){
    /************ CAN Packet Read ************ 
    if (CAN.parsePacket()) {
        if (CAN.packetId() == 0x403) {
            while ( CAN.available() ) {
                motor_collector[index] = CAN.read();
                index++;
            }
        }
    }
    *****************************************/


    /****************** Encoder Signal ******************/ 
    pot_signal_raw = analogRead(EncoderPin);
    encoder_degree = map(pot_signal_raw, 0, 1023, 0, 3600);

    buffer[buffer_index++] = encoder_degree;
    buffer_index = (buffer_index > BUFFER_SIZE ? 0 : buffer_index);
    buffer_average = buffer_avg(buffer, BUFFER_SIZE);

    /*  DELETE AFTER MAKING POT RELIABLE */
    //buffer_average = 1800;
    current_position = buffer_average;

    
    /******************** Debug Topic ******************/ 
    /* rpm value */
    pot_data.x = rpm.data;
    /* pot value */
    pot_data.y = buffer_average;
    /* desired joystick or autonomous steering position */
    pot_data.z = desired_pos;
    pub.publish(&pot_data);
    
    nh.spinOnce();
    delay(2);
}




//////
/*
End of File
*/
//////
