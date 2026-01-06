#include <SFE_BMP180.h>
#include <Wire.h>

// Create an SFE_BMP180 object called "pressure":
SFE_BMP180 pressure;

// Altitude of SparkFun's HQ in Boulder, CO. in meters
// CHANGE THIS to your current city's altitude for accurate readings!
#define ALTITUDE 1655.0 

void setup() {
  Serial.begin(9600);
  Serial.println("REBOOT");

  // Initialize the sensor
  if (pressure.begin())
    Serial.println("BMP180 init success");
  else {
    // Oops, something went wrong, usually a connection problem
    Serial.println("BMP180 init fail\n\n");
    while(1); // Pause forever.
  }
}

void loop() {
  char status;
  double T, P, p0, a;

  Serial.println();
  Serial.print("provided altitude: ");
  Serial.print(ALTITUDE, 0);
  Serial.print(" meters, ");
  Serial.print(ALTITUDE * 3.28084, 0);
  Serial.println(" feet");

  // Step 1: Start a temperature measurement
  status = pressure.startTemperature();
  if (status != 0) {
    
    // Wait for the measurement to complete:
    delay(status);

    // Step 2: Retrieve the completed temperature measurement
    status = pressure.getTemperature(T);
    if (status != 0) {
      // Print out the measurement:
      Serial.print("temperature: ");
      Serial.print(T, 2);
      Serial.print(" deg C, ");
      Serial.print((9.0 / 5.0) * T + 32.0, 2);
      Serial.println(" deg F");

      // Step 3: Start a pressure measurement
      // Parameter is oversampling setting (0 to 3)
      status = pressure.startPressure(3);
      if (status != 0) {
        
        // Wait for the measurement to complete:
        delay(status);

        // Step 4: Retrieve the completed pressure measurement
        status = pressure.getPressure(P, T);
        if (status != 0) {
          // Print absolute pressure:
          Serial.print("absolute pressure: ");
          Serial.print(P, 2);
          Serial.print(" mb, ");
          Serial.print(P * 0.0295333727, 2);
          Serial.println(" inHg");

          // Step 5: Calculate sea-level compensated pressure
          p0 = pressure.sealevel(P, ALTITUDE); 
          Serial.print("relative (sea-level) pressure: ");
          Serial.print(p0, 2);
          Serial.print(" mb, ");
          Serial.print(p0 * 0.0295333727, 2);
          Serial.println(" inHg");

          // Step 6: Calculate altitude based on pressure
          a = pressure.altitude(P, p0);
          Serial.print("computed altitude: ");
          Serial.print(a, 0);
          Serial.print(" meters, ");
          Serial.print(a * 3.28084, 0);
          Serial.println(" feet");
        } else Serial.println("error retrieving pressure measurement\n");
      } else Serial.println("error starting pressure measurement\n");
    } else Serial.println("error retrieving temperature measurement\n");
  } else Serial.println("error starting temperature measurement\n");

  delay(5000); // Pause for 5 seconds.
}
