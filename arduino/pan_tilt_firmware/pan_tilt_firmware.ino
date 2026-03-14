// ============================================================
//  Pan-Tilt Tracker Firmware — Arduino Mega 2560
//  Serial protocol (115200 baud):
//    PC -> Arduino: V<pan_vel>,<tilt_vel>\n   (steps/sec, float)
//                  E0\n  -- enable motors
//                  E1\n  -- disable motors
//                  S\n   -- emergency stop
//                  P\n   -- ping
//                  Z\n   -- zero position counters
//                  G1\n  -- piezo tone ON (4 kHz)
//                  G0\n  -- piezo tone OFF
//                  C\n   -- calibrate tilt upper limit (marks current pos)
//                  B\n   -- calibrate tilt lower limit (marks current pos)
//                  L<upper>,<lower>\n -- set tilt limits from known offsets
//                  Q\n   -- query tilt position & limits
//    Arduino -> PC: OK\n  -- command ack
//                  PONG\n -- ping reply
//                  T<pos>,<upper>,<upperCal>,<lower>,<lowerCal>\n -- tilt info
//
//  DRV8825 NOTES:
//    * Minimum STEP pulse width = 1.9 us. AccelStepper default is 1 us
//      which DRV8825 ignores, causing missed steps ("barely moving").
//      Fix: setMinPulseWidth(2) on each stepper (done below in setup).
//    * ENABLE pin is active LOW (same as A4988).
//    * Microstepping is set by the physical MODE0/MODE1/MODE2 pins on
//      the breakout board (jumpers / pull-ups). The Arduino does NOT
//      need to drive those pins unless you want runtime switching.
//      Set MICROSTEP_MULTIPLIER in config.py to match your board jumpers:
//        All LOW  -> 1  (full step)
//        M0 HIGH  -> 2  (half step)
//        M1 HIGH  -> 4  (1/4)
//        M0+M1    -> 8  (1/8)   <-- recommended
//        M2 HIGH  -> 16 (1/16)
//        M0+M2    -> 32 (1/32)
// ============================================================

#include <AccelStepper.h>

// ===== PIN CONFIGURATION =====
#define PAN_STEP_PIN    3
#define PAN_DIR_PIN     2
#define TILT_STEP_PIN   9
#define TILT_DIR_PIN    8
#define ENABLE_PIN      7   // Shared enable, active LOW (DRV8825 compatible)
#define SIGNAL_PIN     12   // General-purpose output: HIGH when PC sends G1, LOW for G0
// =============================

// ===== MOTOR CONFIGURATION =====
#define PAN_MAX_SPEED     15000  // steps/sec ceiling (~562 RPM at 1/8 step)
#define PAN_ACCELERATION  20000  // steps/sec^2
#define TILT_MAX_SPEED    8000
#define TILT_ACCELERATION 12000

// Commands below this threshold (steps/sec) are treated as zero.
// Prevents detection noise from generating tiny coil pulses when the
// camera should be holding still.
#define PAN_DEADBAND_SPEED   150   // steps/sec  (~6 deg/sec camera)
#define TILT_DEADBAND_SPEED  100
// ================================

// ===== WATCHDOG =====
#define WATCHDOG_TIMEOUT_MS  500    // ms without a V command -> stop
// ====================

// ===== SERIAL BUFFER =====
#define SERIAL_BUF_SIZE      64
// =========================

AccelStepper panStepper (AccelStepper::DRIVER, PAN_STEP_PIN,  PAN_DIR_PIN);
AccelStepper tiltStepper(AccelStepper::DRIVER, TILT_STEP_PIN, TILT_DIR_PIN);

static char    serialBuf[SERIAL_BUF_SIZE];
static uint8_t serialBufLen = 0;

static bool motorsEnabled   = false;
static bool watchdogTripped = false;
static unsigned long lastCmdMs = 0;

static float targetPanSpeed   = 0.0f;
static float targetTiltSpeed  = 0.0f;
static float currentPanSpeed  = 0.0f;
static float currentTiltSpeed = 0.0f;
static unsigned long lastRampUs = 0;

// Piezo buzzer (software toggle — avoids tone() which hijacks Timer 2 / pin 9)
static bool piezoActive       = false;
static unsigned long piezoLastToggleUs = 0;
static const unsigned long PIEZO_HALF_PERIOD_US = 125;  // 4 kHz = 250 us period

// Tilt limits (manual calibration per session)
static bool tiltUpperCalibrated = false;
static bool tiltLowerCalibrated = false;
static long tiltUpperLimit      = 0;
static long tiltLowerLimit      = 0;

// Pan limits
static bool panLimitsCalibrated = false;
static long panUpperLimit       = 0;
static long panLowerLimit       = 0;

// ---------- helpers ----------

// Step current toward target by at most maxStep.
float moveToward(float current, float target, float maxStep) {
    if (current < target) return min(current + maxStep, target);
    if (current > target) return max(current - maxStep, target);
    return target;
}

void enableMotors(bool en) {
    motorsEnabled = en;
    digitalWrite(ENABLE_PIN, en ? LOW : HIGH);
}

void emergencyStop() {
    targetPanSpeed   = 0.0f;
    targetTiltSpeed  = 0.0f;
    currentPanSpeed  = 0.0f;
    currentTiltSpeed = 0.0f;
    panStepper.setSpeed(0.0f);
    tiltStepper.setSpeed(0.0f);
    enableMotors(false);
}

// ---------- serial command processing ----------

void processCommand(const char* cmd) {
    lastCmdMs = millis();
    watchdogTripped = false;

    if (cmd[0] == 'V') {
        const char* comma = strchr(cmd + 1, ',');
        if (comma == NULL) return;
        float p = atof(cmd + 1);
        float t = atof(comma + 1);
        targetPanSpeed  = (abs(p) >= PAN_DEADBAND_SPEED)  ? p : 0.0f;
        targetTiltSpeed = (abs(t) >= TILT_DEADBAND_SPEED) ? t : 0.0f;
        return;
    }

    if (cmd[0] == 'E') {
        if (cmd[1] == '0') {
            enableMotors(true);
            delay(50);
        } else if (cmd[1] == '1') {
            targetPanSpeed   = 0.0f;
            targetTiltSpeed  = 0.0f;
            currentPanSpeed  = 0.0f;
            currentTiltSpeed = 0.0f;
            panStepper.setSpeed(0.0f);
            tiltStepper.setSpeed(0.0f);
            enableMotors(false);
        }
        Serial.print("OK\n");
        return;
    }

    if (cmd[0] == 'S') {
        emergencyStop();
        Serial.print("OK\n");
        return;
    }

    if (cmd[0] == 'P') {
        Serial.print("PONG\n");
        return;
    }

    if (cmd[0] == 'Z') {
        panStepper.setCurrentPosition(0);
        tiltStepper.setCurrentPosition(0);
        tiltUpperCalibrated = false;  // position reference gone
        tiltLowerCalibrated = false;
        panLimitsCalibrated = false;
        Serial.print("OK\n");
        return;
    }

    if (cmd[0] == 'C') {
        tiltUpperLimit = tiltStepper.currentPosition();
        tiltUpperCalibrated = true;
        Serial.print("OK\n");
        return;
    }

    if (cmd[0] == 'B') {
        tiltLowerLimit = tiltStepper.currentPosition();
        tiltLowerCalibrated = true;
        Serial.print("OK\n");
        return;
    }

    if (cmd[0] == 'Q') {
        // Report pan & tilt positions and calibrated limits
        // Format: T<tiltPos>,<tiltUpper>,<tiltUpperCal>,<tiltLower>,<tiltLowerCal>,
        //           <panPos>,<panUpper>,<panLower>,<panCal>
        long tPos = tiltStepper.currentPosition();
        long pPos = panStepper.currentPosition();
        Serial.print("T");
        Serial.print(tPos);    Serial.print(",");
        Serial.print(tiltUpperLimit);  Serial.print(",");
        Serial.print(tiltUpperCalibrated ? 1 : 0); Serial.print(",");
        Serial.print(tiltLowerLimit);  Serial.print(",");
        Serial.print(tiltLowerCalibrated ? 1 : 0); Serial.print(",");
        Serial.print(pPos);    Serial.print(",");
        Serial.print(panUpperLimit);   Serial.print(",");
        Serial.print(panLowerLimit);   Serial.print(",");
        Serial.print(panLimitsCalibrated ? 1 : 0);
        Serial.print("\n");
        return;
    }

    if (cmd[0] == 'L') {
        // Set pan + tilt limits relative to current positions
        // Format: L<panUpperOff>,<panLowerOff>,<tiltUpperOff>,<tiltLowerOff>
        const char* p1 = cmd + 1;
        const char* c1 = strchr(p1, ',');   if (!c1) return;
        const char* c2 = strchr(c1+1, ','); if (!c2) return;
        const char* c3 = strchr(c2+1, ','); if (!c3) return;
        long panPos  = panStepper.currentPosition();
        long tiltPos = tiltStepper.currentPosition();
        panUpperLimit  = panPos  + atol(p1);
        panLowerLimit  = panPos  + atol(c1+1);
        tiltUpperLimit = tiltPos + atol(c2+1);
        tiltLowerLimit = tiltPos + atol(c3+1);
        panLimitsCalibrated = true;
        tiltUpperCalibrated = true;
        tiltLowerCalibrated = true;
        Serial.print("OK\n");
        return;
    }

    if (cmd[0] == 'G') {
        piezoActive = (cmd[1] == '1');
        if (!piezoActive) digitalWrite(SIGNAL_PIN, LOW);
        Serial.print("OK\n");
        return;
    }
}

void readSerial() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialBufLen > 0) {
                serialBuf[serialBufLen] = '\0';
                processCommand(serialBuf);
                serialBufLen = 0;
            }
        } else {
            if (serialBufLen < SERIAL_BUF_SIZE - 1) {
                serialBuf[serialBufLen++] = c;
            }
        }
    }
}

// ---------- watchdog ----------

void checkWatchdog() {
    if (motorsEnabled && !watchdogTripped) {
        if ((millis() - lastCmdMs) > WATCHDOG_TIMEOUT_MS) {
            watchdogTripped = true;
            targetPanSpeed  = 0.0f;
            targetTiltSpeed = 0.0f;
        }
    }
    if (watchdogTripped && motorsEnabled) {
        if (currentPanSpeed == 0.0f && currentTiltSpeed == 0.0f) {
            enableMotors(false);
        }
    }
}

// ---------- setup / loop ----------

void setup() {
    Serial.begin(115200);

    pinMode(ENABLE_PIN,  OUTPUT);
    digitalWrite(ENABLE_PIN,  HIGH);  // de-energized at startup

    pinMode(SIGNAL_PIN,  OUTPUT);
    digitalWrite(SIGNAL_PIN,  LOW);   // always start LOW

    panStepper.setMaxSpeed(PAN_MAX_SPEED);
    panStepper.setSpeed(0.0f);
    panStepper.setMinPulseWidth(2);   // DRV8825 requires >= 1.9 us STEP pulse

    tiltStepper.setMaxSpeed(TILT_MAX_SPEED);
    tiltStepper.setSpeed(0.0f);
    tiltStepper.setMinPulseWidth(2);

    lastCmdMs  = millis();
    lastRampUs = micros();
}

void loop() {
    readSerial();
    checkWatchdog();

    // Software velocity ramp — smooth acceleration toward the PC-commanded speed.
    unsigned long nowUs = micros();
    float dt = (float)(nowUs - lastRampUs) * 1e-6f;
    lastRampUs = nowUs;
    if (dt > 0.05f) dt = 0.05f;  // cap at 50 ms to avoid jumps after pauses

    float panStep  = PAN_ACCELERATION  * dt;
    float tiltStep = TILT_ACCELERATION * dt;
    currentPanSpeed  = moveToward(currentPanSpeed,  targetPanSpeed,  panStep);
    currentTiltSpeed = moveToward(currentTiltSpeed, targetTiltSpeed, tiltStep);

    // Pan limits: block velocity in the clamped direction
    if (panLimitsCalibrated) {
        long panPos = panStepper.currentPosition();
        if (panPos >= panUpperLimit && currentPanSpeed > 0.0f) {
            currentPanSpeed = 0.0f;
            if (targetPanSpeed > 0.0f) targetPanSpeed = 0.0f;
        }
        if (panPos <= panLowerLimit && currentPanSpeed < 0.0f) {
            currentPanSpeed = 0.0f;
            if (targetPanSpeed < 0.0f) targetPanSpeed = 0.0f;
        }
    }

    // Tilt limits: block velocity in the clamped direction
    long tiltPos = tiltStepper.currentPosition();
    if (tiltUpperCalibrated && tiltPos >= tiltUpperLimit && currentTiltSpeed > 0.0f) {
        currentTiltSpeed = 0.0f;
        if (targetTiltSpeed > 0.0f) targetTiltSpeed = 0.0f;
    }
    if (tiltLowerCalibrated && tiltPos <= tiltLowerLimit && currentTiltSpeed < 0.0f) {
        currentTiltSpeed = 0.0f;
        if (targetTiltSpeed < 0.0f) targetTiltSpeed = 0.0f;
    }

    if (motorsEnabled) {
        panStepper.setSpeed(currentPanSpeed);
        tiltStepper.setSpeed(currentTiltSpeed);
    }
    panStepper.runSpeed();
    tiltStepper.runSpeed();

    // Software piezo toggle (no tone() — avoids Timer 2 conflict with pin 9)
    if (piezoActive) {
        if ((nowUs - piezoLastToggleUs) >= PIEZO_HALF_PERIOD_US) {
            piezoLastToggleUs = nowUs;
            digitalWrite(SIGNAL_PIN, !digitalRead(SIGNAL_PIN));
        }
    }
}
