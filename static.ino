 /*
    @year:        2020/2021
    @author:      Sekomer
    @touch:       aksoz19@itu.edu.tr
*/


/* Essential Header Files */
#include    <ros.h>
#include    <CAN.h> 
#include    <string.h>
#include    <rosserial_arduino/Adc.h>


/* Preprocessing */
#define     RADIUS              0.28
#define     BUFFER_SIZE         32
#define     MAX_RPM             3500
#define     MAX_REVERSE_RPM    -2000

/* GEARS */
#define     NEUTRAL    0
#define     FORWARD    1
#define     REVERSE    2

/* CAN IDS */
#define     BRAKE_ID           0x705
#define     STEER_ID           0x700
#define     MOTOR_ID           0x501
#define     MOTOR_ODOM_ID      0x403
#define     BATTERY_INFO_ID    0x402

#define     RPM_MODE           0
#define     CURRENT_MODE       1

/**
*   DATA TYPES FOR CAN COMMUNICATION
*/
typedef union { 
    float   data; 
    uint8_t data_u8[4];
} CAN_Float;


typedef union { 
    float   data[2];
    uint8_t data_u8[8];
} CAN_odom;


typedef union { 
    uint16_t data_u16[2];
    uint8_t  data_u8[4]; 
} STEER;


typedef union {
    uint16_t data;
    struct {
      unsigned int direction : 1;
      unsigned int brake_motor_speed : 5;
      unsigned int regen : 10;
    } bits;
} REGEN_BRAKE;



/* Function Declerations */
int32_t buffer_avg(int32_t *buffer, int32_t size);
float   radian2degree(float input);
int32_t kmh2rpm (float vel);
void    sekomerizasyon(CAN_Float &);


/* CAN variable decleartions */
static CAN_odom speed_odom;
static CAN_odom battery_odom;
static CAN_Float rpm;
static CAN_Float current;

// for braking // 
static STEER recep_tayyip_ercetin;

/* speed, steering and condition info */
static STEER     steering_obj;
static int32_t   raw_steer;
static float     regen = 0;
int   current_position = 0;


/* Encoder Variables */
static int32_t desired_pos;
static int32_t pot_odom;
static int32_t pot_sum;

bool turning_right = false;
bool turning_left = false;


/* Constant Steering Motor Speeds */
static int32_t  const   max_steer_speed  = 10;   // change STM side 
static int32_t  const   min_steer_speed  = 6;
static int32_t  const   high_steer_speed = 4;  // @deprecated


/*  */
static volatile int32_t  GEAR = NEUTRAL;
static volatile int32_t  steer_speed;            // speed variable CAN
static volatile int32_t change_value = 0;       // momentary change in steering angle
static volatile int32_t  AUTONOMOUS = 0;
static volatile int32_t  EXTRA;

static int32_t           brake_speed;
static int32_t           brake_direction;
REGEN_BRAKE              regen_brake_packet;

/* Encoder Buffer Variables */
static int32_t           buffer_index = 0;
static int32_t           buffer[BUFFER_SIZE];
static int32_t           buffer_average = 0;
static const int32_t     EncoderPin = A0;
static int32_t           index = 0;
/**/

/* Debug and Log */
static rosserial_arduino::Adc pot_data;


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


/*
*     ROS CALLBACK  
*/
static int32_t cetin;
bool _debug = 0;

void RosCallback(const rosserial_arduino::Adc &mahmut){
    /* 
     *  @param mahmut : Adc => Data sent from ROS to CANBUS
     *  
     *      * mahmut.adc0 : uint16 => Speed Data
     *      * mahmut.adc1 : uint16 => Angle Data
     *      * mahmut.adc2 : uint16 => Regen Data
     *      * mahmut.adc3 : uint16 => GEAR  Data
     *      * mahmut.adc4 : uint16 => Auto  Data
     *      * mahmut.adc5 : uint16 => Mode  Data
    */
    
    raw_steer  = mahmut.adc1;
    regen      = mahmut.adc2;
    GEAR       = mahmut.adc3;
    AUTONOMOUS = mahmut.adc4;
    EXTRA      = mahmut.adc5;

    
    /******************** Speed Logic ******************/ 
    if (mahmut.adc5 == RPM_MODE) {
      switch(GEAR) {
          case FORWARD:
              //rpm.data = map((long) mahmut.adc0, 0, 1000, 0, MAX_RPM); 
              rpm.data = mahmut.adc0;
              current.data = 1;
              break;
          case REVERSE:
              //rpm.data = map((long) mahmut.adc0, 0, 1000, 0, MAX_REVERSE_RPM);
              rpm.data = mahmut.adc0;
              current.data = 0.8;
              break;
          case NEUTRAL:
              rpm.data = 0;
              current.data = 0;
              break;    
      }
    }
    // current mode
    else if (mahmut.adc5 == CURRENT_MODE) {
     switch(GEAR) {
          case FORWARD:
              current.data = float(mahmut.adc0) / 1000.0;
              rpm.data = 20000;
              break;
          case REVERSE:
              current.data = float(mahmut.adc0) / 1000.0;
              rpm.data = -20000;
              break;
          case NEUTRAL:
              current.data = 0;
              rpm.data = 0;
              break;    
      }   
    }
    else {
      rpm.data = 0;
      current.data = 0;
    }
    

    if (rpm.data == 0)
        current.data = 0.0;
    if (regen) {
        rpm.data = 0; /* security */
        current.data = regen / 1000;
    }

    /* BRAKING */
    /******************************************/
    regen_brake_packet.data = mahmut.adc2;
    brake_speed = regen_brake_packet.bits.brake_motor_speed;
    brake_direction = regen_brake_packet.bits.direction;
    //regen = regen_brake_packet.bits.regen;

    recep_tayyip_ercetin.data_u16[0] = brake_direction;
    recep_tayyip_ercetin.data_u8[2] = brake_speed;

    if (brake_speed != 0) {
        recep_tayyip_ercetin.data_u16[0] = brake_direction;
        recep_tayyip_ercetin.data_u8[2] = brake_speed;    
        /* security */
        //rpm.data = 0;
        //current.data = 0;
    }
    else {
        recep_tayyip_ercetin.data_u16[0] = 0;
        recep_tayyip_ercetin.data_u8[2] = 0;
    }
    
    /***************************************************/

    
    sekomerizasyon(current);
    
    /******************** Steering Logic ******************/ 
    /*  
        to send data from terminal or to read data from ROS
        desired_pos = mahmut.adc1
        to read data from pot add a potansiometer to A1 pin
        desired_pos = direksiyon_pot (A1)
    */

    /* TAM TERSİYSE */
    
    desired_pos = mahmut.adc1;
    change_value = (desired_pos - current_position); // get change value
 

    /* Speed Control */
    /*
    if (abs(change_value) < 75)
        steer_speed = low_steer_speed;
    else if (change_value < 350 && change_value > -350)
        steer_speed = high_steer_speed;
    else
        steer_speed = max_steer_speed;
    */

    cetin = abs(change_value);
    if (cetin > 1800)
      cetin = 1800;
      
    steer_speed = map(cetin, 0, 1800, min_steer_speed, max_steer_speed);

    /*fren experimental*/
    if (regen) {
        recep_tayyip_ercetin.data_u16[0] = 0;
        recep_tayyip_ercetin.data_u8[2] = 8;
    }
    else {
        recep_tayyip_ercetin.data_u16[0] = 1;
        recep_tayyip_ercetin.data_u8[2] = 8;
    }

    
    /*
        @steering_obj.data_u16[0]  => steering speed
        @steering_obj.data_u8[2]   => steering 
    */

    if (change_value > 0) {
        steering_obj.data_u16[0] = steer_speed;
        steering_obj.data_u8[2]  = 1;      // left steer direction
        _debug = 1;
    }
    else if (change_value < 0) {
        steering_obj.data_u16[0] = steer_speed;
        steering_obj.data_u8[2]  = 0;      // right steer direction
        _debug = 0;
    }
    else 
        steering_obj.data_u16[0]  = 0;

    /*
     * Driving Motor Packet
    */
    /* rpm */
    CAN.beginPacket(MOTOR_ID);
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
    CAN.beginPacket(STEER_ID);
    CAN.write(steering_obj.data_u8[0]);
    CAN.write(steering_obj.data_u8[1]);
    CAN.write(steering_obj.data_u8[2]);
    CAN.endPacket();

    /*
    CAN.beginPacket(BRAKE_ID);
    CAN.write(recep_tayyip_ercetin.data_u8[0]);
    CAN.write(recep_tayyip_ercetin.data_u8[1]);
    CAN.write(recep_tayyip_ercetin.data_u8[2]);
    CAN.endPacket();
    */
}   




/* 
 *  IT DOESN'T WORK WITH A QUEUE VALUE
 *  @future_debug
*/

/******************** Creating ROS Node ******************/ 
static ros::NodeHandle nh;
static ros::Subscriber<rosserial_arduino::Adc> sub("/seko", &RosCallback);
static ros::Publisher pub("pot_topic", &pot_data);



/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

void sekomerizasyon(CAN_Float &current)
{
    if (current.data > 1.0)
        current.data = 1.0;
    else if (current.data < 0.0)
        current.data = 0.0;
}

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
*   @deprecated
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
    pinMode(A1, INPUT);
    memset(buffer, 0, BUFFER_SIZE * sizeof(int));
    Serial.begin(57600);

    /****************** ROS Initialization ******************/ 
    nh.getHardware()->setBaud(57600);
    nh.initNode();
    nh.subscribe(sub);
    nh.advertise(pub);

    /* security */
    rpm.data = 0;
    current.data = 0;
    steering_obj.data_u16[0]  = 0;
    
CAN_INIT:
    if(CAN.begin(1E6)) {
      ;/* maybe wait */;
    } 
    else {
        goto CAN_INIT;
        Serial.println(31);
    }

    /* safety */
    /* rpm */
    CAN.beginPacket(MOTOR_ID);
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
    /* Steering Motor Packet */
    CAN.beginPacket(STEER_ID);
    CAN.write(steering_obj.data_u8[0]);
    CAN.write(steering_obj.data_u8[1]);
    CAN.write(steering_obj.data_u8[2]);
    CAN.endPacket();
}


/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/*
*   Arduino Loop
*
*   rosrun rosserial_arduino serial_node.py _port:=/dev/ttyXXXn _baud:=57600
*/
void loop(){
    /************ CAN Packet Read ************/ 
    if (CAN.parsePacket()) {
        if (CAN.packetId() == MOTOR_ODOM_ID) {
            while (CAN.available())
              speed_odom.data_u8[index++] = CAN.read();
        }
        else if (CAN.packetId() == BATTERY_INFO_ID)
          while (CAN.available())
              battery_odom.data_u8[index++] = CAN.read();
        index ^= index;
    }

    
    /****************** Encoder Signal ******************/ 
    /*
     * Elder Version, Unnecessary sums
     */
     /* 
    int pot_signal_raw = analogRead(EncoderPin);
    int encoder_degree = map(pot_signal_raw, 0, 1023, 0, 3600);
    
    buffer[buffer_index++] = encoder_degree;
    buffer_average = buffer_avg(buffer, BUFFER_SIZE);
    buffer_index = (buffer_index > BUFFER_SIZE ? 0 : buffer_index);
    */   


    pot_sum -= buffer[buffer_index];
    int32_t analog_read = analogRead(EncoderPin);
    buffer[buffer_index] = map(analog_read, 0, 1023, 0, 3600);
    pot_sum += buffer[buffer_index];
    buffer_index++;
    buffer_index = (buffer_index >= BUFFER_SIZE ? 0 : buffer_index);

    buffer_average = pot_sum / BUFFER_SIZE;
    pot_odom = buffer_average;

   
    /* DRIVING MODE */
    if (! AUTONOMOUS)
        buffer_average = 1800;
    
    current_position = buffer_average;
    
    
    /******************** Debug Topic ******************/ 
    /* rpm */
    pot_data.adc0 = speed_odom.data[1];
    /* steer */
    pot_data.adc1 = pot_odom;
    /* bus voltage / bus current */
    pot_data.adc2 = battery_odom.data[0];
    pot_data.adc3 = desired_pos;
    /* unused */
    pot_data.adc4 = current_position;
    pot_data.adc5 = (change_value);
    
    pub.publish(&pot_data);
    nh.spinOnce();
}


//////
/*
End of File
*/
//////
