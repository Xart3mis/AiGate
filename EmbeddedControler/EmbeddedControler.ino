#include <Servo.h>
#include <Wire.h>

Servo horizontal;
Servo vertical;

unsigned long previousMillis = 0;

int h = 90;
int v = 90;

int x = 0;
int y = 0;

void setup() {
  Wire.begin(8);
  Wire.onReceive(receiveEvent);

  vertical.attach(5);
  horizontal.attach(3);
  
  horizontal.write(h);
  vertical.write(v);
  Serial.begin(9600);
}

void receiveEvent(int howMany) {
  int data[howMany];

  for (int i = 0; i < howMany; i++) {
    data[i] = Wire.read();
  }

  x = data[1];
  y = data[2];

  for (int i = 1; i < howMany; i++) {
    //Serial.print(data[i]); Serial.print(" ");
  }
  Serial.println();
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= 80) {
    previousMillis = currentMillis;

    switch (x) {
      case 0: {
          break;
        }
      case 1: {
          h++;
          if (h >= 130) {
            h = 130;
          }
          break;
        }
      case 2: {
          h--;
          if (h <= 50) {
            h = 50;
          }
          break;
        }
      case 3: {
          if (h > 95) {
            h--;
          }
          else if (h < 85) {
            h++;
          }
          break;
        }
    }
    horizontal.write(h);

    switch (y) {
      case 0: {
          break;
        }
      case 1: {
          v++;
          if (v >= 170) {
            v = 170;
          }
          break;
        }
      case 2: {
          v--;
          if (v <= 10) {
            v = 10;
          }
          break;
        }
      case 3: {
          if (v > 95) {
            v--;
          }
          else if (v < 85) {
            v++;
          }
          break;
        }
    }
    vertical.write(v);

    //Serial.print(x); Serial.print(" "); Serial.println(y);
  }
}
