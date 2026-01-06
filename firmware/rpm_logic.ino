/*
 * HVAC Sentinel - IR RPM Sensor Module
 * Monitors fan speed using Timer1 interrupts and direct register manipulation.
 * Hardware: Arduino Nano/Uno with IR Module on Interrupt Pin 0 (Digital Pin 2).
 */

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>

volatile unsigned long rpmtime = 0; // "volatile" is safer for ISR variables
float rpmfloat;
unsigned int rpm;
volatile bool tooslow = 1;

void setup()
{
  Serial.begin(9600);
  
  // Timer1 Configuration for Timeout Detection
  TCCR1A = 0;
  TCCR1B = 0;
  TCCR1B |= (1 << CS12);    // Set Prescaler to 256
  TIMSK1 |= (1 << TOIE1);   // Enable Timer Overflow Interrupt
  
  pinMode(2, INPUT); 
  attachInterrupt(0, RPM, FALLING); // Interrupt on Sensor Falling Edge
}

// Timer1 Overflow Interrupt (Fan is stopped/too slow)
ISR(TIMER1_OVF_vect)
{
  tooslow = 1;
}

void loop()
{
  delay(1000); // Update every second
  
  if (tooslow == 1) {
    Serial.println("Status: Fan Stopped / Too Slow");
  }
  else {
    // Calculate RPM based on timer ticks (Prescaler 256 logic)
    rpmfloat = 120 / (rpmtime / 31250.00); 
    rpm = round(rpmfloat);
    
    Serial.print("RPM = ");
    Serial.println(rpm);
  }
}

// Sensor Interrupt Routine
void RPM ()
{
  rpmtime = TCNT1; // Capture timer value
  TCNT1 = 0;       // Reset timer
  tooslow = 0;     // Clear timeout flag
}
