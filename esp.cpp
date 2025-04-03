#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "Adafruit_MAX30102.h"

// Configuración WiFi
const char* ssid = "NOMBRE_WIFI";
const char* password = "CONTRASEÑA_WIFI";

// Servidor FastAPI(Por definir aun)
const char* serverUrl = "http://192.168.1.100:8000/data";  

// Pines GSR
#define GSR_PIN 34

// Inicialización del sensor MAX30102
Adafruit_MAX30102 max30102;

void setup() {
  Serial.begin(115200);
  
  // Conectar a WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Conectando a WiFi...");
  }
  Serial.println("Conectado a WiFi");

  // Inicializar I2C para MAX30102
  Wire.begin();
  if (!max30102.begin()) {
    Serial.println("MAX30102 no detectado. Verifica la conexión.");
    while (1);
  }
  Serial.println("MAX30102 iniciado correctamente.");
}

void loop() {

  int gsrValue = analogRead(GSR_PIN);
  
  int bpm = 0, spo2 = 0;
  uint32_t ir, red;

  if (max30102.getIR() > 50000) { 
    ir = max30102.getIR();
    red = max30102.getRed();

    bpm = map(ir, 50000, 100000, 60, 120);
    spo2 = map(red, 30000, 100000, 95, 100);
  } else {
    bpm = 0;
    spo2 = 0;
  }

  Serial.printf("GSR: %d, BPM: %d, SpO2: %d\n", gsrValue, bpm, spo2);

  // Enviar datos al servidor
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    // Crear JSON con los datos
    String jsonData = "{\"gsr\": " + String(gsrValue) + 
                      ", \"bpm\": " + String(bpm) + 
                      ", \"spo2\": " + String(spo2) + "}";

    int httpResponseCode = http.POST(jsonData);
    
    if (httpResponseCode > 0) {
      Serial.println("Datos enviados correctamente.");
    } else {
      Serial.printf("Error al enviar datos: %d\n", httpResponseCode);
    }
    http.end();
  }

  delay(1000);  // Enviar datos cada 1 segundo
}
   