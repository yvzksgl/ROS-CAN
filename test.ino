/*
    @year:        2020/2021
    @author:      Sekomer
    @touch:       aksoz19@itu.edu.tr
*/


/* Essential Header Files */
#include    <ros.h>
#include    <CAN.h>
#include    <Wire.h>
#include    <string.h>
#include    <LiquidCrystal_I2C.h>

#include    <std_msgs/Float32.h>
#include    <geometry_msgs/Twist.h>
#include    <geometry_msgs/Vector3.h>
#include    <rosserial_arduino/Adc.h>


/* Preprocessing */
#define     RADIUS              0.28
#define     BUFFER_SIZE         32
#define     GEAR_SCALE          7
#define     MAX_STEERING_ANGLE  35
#define     MAX_RPM             350
#define     MAX_REVERSE_RPM    -100

/* LCD */
#define     ASCII                    48
#define     LCD_AUTONOMOUS            2
#define     LCD_SPEED_FIRST           6
#define     LCD_SPEED_SECOND          7
#define     LCD_CURRENT_FIRST        11
#define     LCD_CURRENT_SECOND       13
#define     LCD_GEAR                  2
#define     LCD_REGEN                 6
#define     LCD_BRAKE                10
#define     LCD_STEERING_FIRST       14
#define     LCD_STEERING_SECOND      15

/* GEARS */
#define     NEUTRAL    0
#define     FORWARD    1
#define     REVERSE    2


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
unsigned int speed_odom_collector[8];
int32_t      odom_speed;
unsigned int steer_collector[8];
unsigned int odom_index = 0;

CAN_Float motor_odometry;
CAN_Float steer_odometry;
CAN_Float current;

/* speed, steering and condition info */
CAN_Float rpm;
STEER     steering_obj;
int32_t   raw_speed;
int32_t   raw_steer;
float     regen;
int32_t   current_position = 0;


/* Encoder Variables */
int32_t pot_signal_raw;
int32_t encoder_degree;
int32_t desired_pos;


/* Constant Steering Motor Speeds */
int32_t  const   max_steer_speed = 14;  // change STM side 
int32_t  const   min_steer_speed  = 4;
int32_t  const   high_steer_speed = 10;  // @deprecated


/*  */
volatile int32_t GEAR = NEUTRAL;
volatile int32_t steer_speed;                // speed variable CAN
volatile int32_t brake;
volatile int32_t change_value = 0;   // momentary change in steering angle
volatile int32_t AUTONOMOUS = 0;
volatile int32_t EXTRA;
int8_t anil = 0;

/* Encoder Buffer Variables */
int32_t         buffer_index = 0;
int32_t         buffer[BUFFER_SIZE];
int32_t         buffer_average = 0;
int32_t         lcd_buffer_average = 0;
const int32_t   EncoderPin = A0;
/**/

/* Debug and Log */
geometry_msgs::Vector3 pot_data;

LiquidCrystal_I2C lcd(0x27, 20, 4);
char first_row[17]  = {'A','=',' ',' ','V','=',' ',' ',' ','C','=',' ','.',' ',' ',' ','\0'};
char second_row[17] = {'G','=',' ',' ','R','=',' ',' ','B','=',' ',' ','S','=',' ',' ','\0'};


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


/*
*     ROS CALLBACK  
*/
void RosCallback(const rosserial_arduino::Adc &mahmut){
    /* 
     *  @param mahmut : Adc => Data sent from ROS to CANBUS
     *  
     *      * mahmut.adc0 : uint16 => Speed Data
     *      * mahmut.adc1 : uint16 => Angle Data
     *      * mahmut.adc2 : uint16 => Regen Data
     *      * mahmut.adc3 : uint16 => GEAR  Data
     *      * mahmut.adc4 : uint16 => Light Data
     *      * mahmut.adc5 : uint16 => SOS   Data
    */
    
    raw_speed  = map(mahmut.adc0, 0, 1000, 0, 99);
    raw_steer  = mahmut.adc1;
    regen      = mahmut.adc2;
    GEAR       = mahmut.adc3;
    AUTONOMOUS = mahmut.adc4;
    EXTRA      = mahmut.adc5;
    
    /******************** Speed Logic ******************/ 

    
    switch(GEAR) {
        case FORWARD:
            rpm.data = map((long) mahmut.adc0, 0, 1000, 0, MAX_RPM); 
            current.data = 0.9;
            break;
        case REVERSE:
            rpm.data = map((long) mahmut.adc0, 0, 1000, 0, MAX_REVERSE_RPM);
            current.data = 0.9;
            break;
        case NEUTRAL:
            rpm.data = 0;
            current.data = 0;
            break;    
    }


    if (rpm.data == 0)
        current.data = 0.0;
    if (regen) {
        rpm.data = 0; /* security */
        current.data = regen / 1000;
    }
        

    
    /******************** Steering Logic ******************/ 
    /*  
        to send data from terminal or to read data from ROS
        desired_pos = mahmut.adc1

        to read data from pot add a potansiometer to A1 pin
        desired_pos = direksiyon_pot (A1)
    */

    /* TAM TERSİYSE */
    
    desired_pos = mahmut.adc1;
    
    
    change_value = desired_pos - current_position; // get change value

    /* Speed Control */
    /*
    if (abs(change_value) < 75)
        steer_speed = low_steer_speed;
    else if (change_value < 350 && change_value > -350)
        steer_speed = high_steer_speed;
    else
        steer_speed = max_steer_speed;
    */
    
    steer_speed = map(abs(change_value), 0, 1800, min_steer_speed, max_steer_speed);
    /*
        @steering_obj.data_u16[0]  => steering speed
        @steering_obj.data_u8[2]   => steering 
    */

    if (change_value > 0) {
        steering_obj.data_u16[0] = steer_speed;
        steering_obj.data_u8[2]  = 1;      // steer direction
    }
    else if (change_value < 0) {
        steering_obj.data_u16[0] = steer_speed;
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
ros::Subscriber<rosserial_arduino::Adc> sub("/seko", &RosCallback);
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

    /* LDC */
    lcd.init();
    lcd.init();
    lcd.backlight();
    
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
    long start_t = millis();
    /************ CAN Packet Read ************/ 
    if (CAN.parsePacket())
        if (CAN.packetId() == 0x403)
            while (CAN.available())
                speed_odom_collector[odom_index++] = CAN.read();
    
    odom_index ^= odom_index;
    
    /****************** Encoder Signal ******************/ 
    pot_signal_raw = analogRead(EncoderPin);
    encoder_degree = map(pot_signal_raw, 0, 1023, 0, 3600);

    buffer[buffer_index++] = encoder_degree;
    buffer_average = buffer_avg(buffer, BUFFER_SIZE);
    lcd_buffer_average = map(buffer_average, 0, 3600, 0, 99);
    buffer_index = (buffer_index > BUFFER_SIZE ? 0 : buffer_index);
    
    int32_t temp = buffer_average;
    
    /* DRIVING MODE */
    if (! AUTONOMOUS)
        buffer_average = 1800;
    
    current_position = buffer_average;

    
    /******************** Debug Topic ******************/ 
    /* rpm value */
    pot_data.x = rpm.data;
    /* pot value */
    pot_data.y = buffer_average;
    /* desired joystick or autonomous steering position */
    pot_data.z = temp;
    pub.publish(&pot_data);


    /*
     *  This part has been moved to another microcontroller
    */

    /*    
    first_row[LCD_AUTONOMOUS]         = AUTONOMOUS + ASCII;
    first_row[LCD_SPEED_FIRST]        = raw_speed / 10 + ASCII; 
    first_row[LCD_SPEED_SECOND]       = raw_speed % 10 + ASCII;
    first_row[LCD_CURRENT_FIRST]      = (int) current.data + ASCII;
    first_row[LCD_CURRENT_SECOND]     = int(current.data * 10.0)%10 + ASCII;
    first_row[15] = (anil++ % 2) ? 'x' : '+';  
    
    second_row[LCD_GEAR]              = (GEAR == 1 ? 'F' : (GEAR == 2 ? 'R' : 'N'));
    second_row[LCD_REGEN]             = map(regen, 0, 1000, 0, 9) + ASCII;
    second_row[LCD_BRAKE]             = map(brake, 0, 1000, 0, 9) + ASCII;
    second_row[LCD_STEERING_FIRST]    = lcd_buffer_average / 10 + ASCII;
    second_row[LCD_STEERING_SECOND]   = lcd_buffer_average % 10 + ASCII;

    "/* This part drop node hertz from ~185 to 17
    lcd.setCursor(0,0);
    lcd.print(first_row);
    lcd.setCursor(0,1);
    lcd.print(second_row);

    */
 
    nh.spinOnce();
}




//////
/*
End of File
*/
//////
