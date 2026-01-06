/*
 * HVAC Sentinel - DHT11 Sensor Module
 * Reads Temperature and Humidity data for the ESP8266 Node.
 */

#include <dht.h>

#define dht_apin A0 // Analog Pin sensor is connected to 
dht DHT;

void setup(){
  Serial.begin(9600);
  delay(500); // Delay to let system boot
  Serial.println("DHT11 Humidity & Temperature Sensor\n\n");
  delay(1000); // Wait before accessing Sensor
}

void loop(){
  // Start of Program 
  DHT.read11(dht_apin);
    
  Serial.print("Current humidity = ");
  Serial.print(DHT.humidity);
  Serial.print("%  ");
  Serial.print("temperature = ");
  Serial.print(DHT.temperature); 
  Serial.println("C  ");
    
  delay(5000); // Wait 5 seconds before accessing sensor again.
  // Fastest should be once every two seconds.
}
