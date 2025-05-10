#include "esp_camera.h"
#include <WiFi.h>
//
// WARNING!!! PSRAM IC required for UXGA resolution and high JPEG quality
//            Ensure ESP32 Wrover Module or other board with PSRAM is selected
//            Partial images will be transmitted if image exceeds buffer size
//

// Select camera model
#define CAMERA_MODEL_WROVER_KIT // Has PSRAM

#include "camera_pins.h"

const char* ssid     = "IZZI-AA74";   //input your wifi name
const char* password = "t6NDhrhCpCrB";   //input your wifi passwords

void startCameraServer();

void setup() {
  Serial.begin(921600);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  // drop down frame size for higher initial frame rate
  s->set_framesize(s, FRAMESIZE_VGA);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
}

unsigned long startTime = 0;
bool sending = false;

void loop() {
  if (!sending) {
    startTime = millis(); // Guardar tiempo de inicio
    sending = true;
  }

  // Si han pasado menos de 5 segundos, sigue enviando
  if (millis() - startTime <= 10000) {
    sendImageToAPI();
    delay(1000); // Esperar 1 segundo entre cada imagen
  } else {
    Serial.println("Envío finalizado.");
    while (true); // Detener ejecución (puedes reiniciar o hacer otra cosa aquí)
  }
}

#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h> // Asegúrate de instalar esta librería desde el Library Manager

void sendImageToAPI() {
  camera_fb_t * fb = esp_camera_fb_get(); // capturar imagen
  if (!fb) {
    Serial.println("Error capturando imagen");
    return;
  }

  HTTPClient http;
  http.begin("http://192.168.0.127:5000/upload"); // Reemplaza con la IP correcta
  http.addHeader("Content-Type", "image/jpeg");

  int httpResponseCode = http.POST(fb->buf, fb->len); // Enviar buffer JPEG

  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("Código de respuesta: %d\n", httpResponseCode);
    Serial.println("Respuesta del servidor:");
    Serial.println(response);

    // Parsear JSON (suponiendo que la respuesta es tipo: {"emotion":"happy",...})
    StaticJsonDocument<256> doc;
    DeserializationError error = deserializeJson(doc, response);

    if (!error) {
      if (doc.containsKey("emotion")) {
        const char* emotion = doc["emotion"];
        float confidence = doc["confidence"];
        int bpm = doc["bpm"];
        float sweat = doc["sweat_level"];

        Serial.printf("Emoción: %s\n", emotion);
        Serial.printf("Confianza: %.2f\n", confidence);
        Serial.printf("BPM: %d\n", bpm);
        Serial.printf("Sudoración: %.2f\n", sweat);
      } else if (doc.containsKey("error")) {
        Serial.print("Error del servidor: ");
        Serial.println(doc["error"].as<const char*>());
      } else {
        Serial.println("Respuesta desconocida del servidor.");
      }
    } else {
      Serial.println("Error al parsear JSON");
    }

  }
  http.end();
  esp_camera_fb_return(fb);
}