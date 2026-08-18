// CHORDS ECG Compatible – Arduino UNO RAW ASCII Streaming

#define EKG A0

const unsigned long sampleRate = 250;       // 250 Hz recommended by CHORDS
const unsigned long samplePeriod = 1000000UL / sampleRate;
unsigned long lastSampleMicros = 0;

void setup() {
  Serial.begin(115200);   
  delay(200);
  Serial.println("StartUp!");
}

void loop() {
  unsigned long now = micros();

  if (now - lastSampleMicros >= samplePeriod) {
    lastSampleMicros += samplePeriod;

    int value = analogRead(EKG);   // 0–1023

    Serial.println(value);         // CHORDS requires ASCII numeric data
  }
}
