/*
    @year:        2020/2021
    @author:      Sekomer
    @touch:       aksoz19@itu.edu.tr
*/

#include "ros.h"
#include "LiquidCrystal_I2C.h"
#include "rosserial_arduino/Adc.h"
#include <EEPROM.h>


#define ASCII   48
#define NEUTRAL  0
#define FORWARD  1
#define REVERSE  2


typedef struct {
    unsigned int steer      : 12;
    unsigned int velocity   : 8;
    unsigned int brake      : 8;
    unsigned int regen      : 4;
    unsigned int current    : 4;
    unsigned int gear       : 2;
    unsigned int autonomous : 1;
} LCD_RAW;


LCD_RAW data;

void RosCallback(const rosserial_arduino::Adc &cb_data){
  data.velocity   = cb_data.adc0;
  data.steer      = cb_data.adc1;
  data.autonomous = cb_data.adc2;
  data.gear       = cb_data.adc3;
  data.regen      = cb_data.adc4;
  data.current    = cb_data.adc5;
}

ros::NodeHandle nh;
ros::Subscriber<rosserial_arduino::Adc> sub("/screen", &RosCallback);

LiquidCrystal_I2C lcd(0x27,20,4);

void setup() {
    Serial.begin(57600);
    /****************** ROS Initialization ******************/ 
    nh.getHardware()->setBaud(57600);
    nh.initNode();
    nh.subscribe(sub);
    delay(100);

  /*
  char first_row[]  = {'A','=',' ',' ','V','=',' ',' ',' ','C','=',' ','.',' '};
  char second_row[] = {'G','=',' ',' ','R','=',' ',' ','B','=',' ',' ','S','=',' ',' '};
  */
  
    EEPROM.update(0, 'A');
    EEPROM.update(1, 'V');
    EEPROM.update(2, 'C');
    EEPROM.update(4, 'G');
    EEPROM.update(5, 'R');
    EEPROM.update(6, 'B');
    EEPROM.update(7, 'S');
    EEPROM.update(8, '=');
    EEPROM.update(9, '.');
    

    lcd.init();
    lcd.backlight();
    lcd.clear();
}


void loop() {
  /* FIRST ROW */

  /* AUTONOMOUS */
  lcd.setCursor(0,0);
  lcd.write(EEPROM.read(0));
  lcd.setCursor(1,0);
  lcd.write(EEPROM.read(8));
  lcd.setCursor(2,0);
  lcd.write(data.autonomous + ASCII);
  

  /* VELOCITY */
  lcd.setCursor(4,0);
  lcd.write(EEPROM.read(1));
  lcd.setCursor(5,0);
  lcd.write(EEPROM.read(8));
  lcd.setCursor(6,0);
  lcd.write(data.velocity / 10 + ASCII);
  lcd.setCursor(7,0);
  lcd.write(data.velocity % 10 + ASCII);
  
  /* CURRENT */
  lcd.setCursor(9,0);
  lcd.write(EEPROM.read(2));
  lcd.setCursor(10,0);
  lcd.write(EEPROM.read(8));
  lcd.setCursor(11,0);
  lcd.write(data.current / 10 + ASCII);
  lcd.setCursor(12,0);
  lcd.write(EEPROM.read(9));
  lcd.setCursor(13,0);
  lcd.write(data.current % 10 + ASCII);
  

  /* SECOND ROW */
  /* GEAR */
  lcd.setCursor(0,1);
  lcd.write(EEPROM.read(4));
  lcd.setCursor(1,1);
  lcd.write(EEPROM.read(8));
  lcd.setCursor(2,1);
  lcd.write(data.gear == 0 ? 'N' : data.gear == 1 ? 'F' : 'R');


  /* BRAKE */
  lcd.setCursor(4,1);
  lcd.write(EEPROM.read(6));
  lcd.setCursor(5,1);
  lcd.write(EEPROM.read(8));
  lcd.setCursor(6,1);
  lcd.write((data.brake / 10) % 10 + ASCII);
  lcd.setCursor(7,1);
  lcd.write(data.brake % 10 + ASCII);

  /* STEERING */
  lcd.setCursor(9,1);
  lcd.write(EEPROM.read(7));
  lcd.setCursor(10,1);
  lcd.write(EEPROM.read(8));
  lcd.setCursor(11,1);
  lcd.write((data.steer / 1000) % 10 + ASCII);
  lcd.setCursor(12,1);
  lcd.write((data.steer / 100) % 10 + ASCII);
  lcd.setCursor(13,1);
  lcd.write((data.steer / 10) % 10 + ASCII);
  lcd.setCursor(14,1);
  lcd.write(data.steer % 10 + ASCII);

  nh.spinOnce();
}

//EOF
