#include <ESP8266WiFi.h>
#include <FirebaseArduino.h>

// --- CONFIGURATION ---
// Replace with your actual Firebase project URL (without "https://")
#define FIREBASE_HOST "YOUR_PROJECT_ID.firebaseio.com" 
// Replace with your Database Secret
#define FIREBASE_AUTH "YOUR_DATABASE_SECRET" 

// Note: This code works for WPA2-Personal (Home WiFi). 
// It will NOT work for "eduroam" (Enterprise) which requires a username.
#define WIFI_SSID "Your_WiFi_Name"
#define WIFI_PASSWORD "Your_WiFi_Password"

float timeSinceLastRead = 0; 

void setup() {
  Serial.begin(9600); 

  // CONNECT TO WIFI
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("connecting");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();
  Serial.print("connected: ");
  Serial.println(WiFi.localIP());
  
  // CONNECT TO FIREBASE
  Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);
}

void loop() {
  // Dummy values for testing (Set these to 1 as requested)
  float h = 1;      // Humidity
  float t = 1;      // Temperature
  float s = 1;      // Air Speed
  float rot = 1;    // Rotation Speed
  float p = 1;      // Power

  // Send data to Firebase
  // Note: These tag names ("Temp", "Humidity") will appear in your database
  Firebase.setFloat("Temp", t); 
  Firebase.setFloat("Humidity", h); 
  Firebase.setFloat("Speed", s); 
  Firebase.setFloat("Rotation", rot); 
  Firebase.setFloat("Power", p); 

  // Check for errors
  if (Firebase.failed()) {
      Serial.print("pushing /logs failed:");
      Serial.println(Firebase.error()); 
      return;
  } else {
      Serial.println("Data sent successfully");
  }

  delay(5000); // Wait 5 seconds before next loop
}
