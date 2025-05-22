#pragma config(Sensor, S1, xTouchSensor, sensorTouch)
#pragma config(Sensor, S2, yTouchSensor, sensorTouch)
#pragma config(Motor, motorA, xGantryMotor, tmotorNXT, PIDControl, encoder)
#pragma config(Motor, motorB, yGantryMotor, tmotorNXT, PIDControl, encoder)

// Constants
const int MAX_WRONG = 6;
const int GANTRY_SPEED = 30;
const int MAX_WORD_SIZE = 10;
const int TIMEOUT_MS = 5000;  // Timeout for motor operations

// Variables
bool abort = false;
int numWrong = 0;
bool gameOver = false;
bool win = false;

char guessedWord[MAX_WORD_SIZE] = {'_', '_', '_', '_', '_', '\0'};
char incorrectGuesses[MAX_WRONG] = {};
char correctGuesses[MAX_WORD_SIZE] = {};
char word[] = "ROBOT";  // Test word
int numRight = 0;

// ? Function: Calibration with timeout & feedback
void calibrate(bool &abort) {
    displayString(0, "Calibrating...");

    long startTime = nSysTime;

    // X Gantry Calibration
    while (SensorValue[xTouchSensor] == 0 && !abort) {
        if (nSysTime - startTime > TIMEOUT_MS) {
            displayString(1, "X Calibration failed (timeout)");
            abort = true;
            break;
        }
        motor[xGantryMotor] = -GANTRY_SPEED;
    }
    motor[xGantryMotor] = 0;

    startTime = nSysTime;

    // Y Gantry Calibration
    while (SensorValue[yTouchSensor] == 0 && !abort) {
        if (nSysTime - startTime > TIMEOUT_MS) {
            displayString(1, "Y Calibration failed (timeout)");
            abort = true;
            break;
        }
        motor[yGantryMotor] = -GANTRY_SPEED;
    }
    motor[yGantryMotor] = 0;

    if (!abort) {
        displayString(1, "Calibration complete");
    }
    wait1Msec(1000);
}

// ? Function: Move gantries with encoder validation
void moveGantry(int xDist, int yDist) {
    const int ERROR_MARGIN = 5;

    nMotorEncoder[xGantryMotor] = 0;
    nMotorEncoder[yGantryMotor] = 0;

    motor[xGantryMotor] = (xDist > 0) ? GANTRY_SPEED : -GANTRY_SPEED;
    motor[yGantryMotor] = (yDist > 0) ? GANTRY_SPEED : -GANTRY_SPEED;

    long startTime = nSysTime;

    while ((abs(nMotorEncoder[xGantryMotor]) < abs(xDist) - ERROR_MARGIN ||
            abs(nMotorEncoder[yGantryMotor]) < abs(yDist) - ERROR_MARGIN) &&
           (nSysTime - startTime < TIMEOUT_MS)) {
        // Display position feedback
        displayString(3, "X: %d, Y: %d", nMotorEncoder[xGantryMotor], nMotorEncoder[yGantryMotor]);
    }

    motor[xGantryMotor] = 0;
    motor[yGantryMotor] = 0;

    if (nSysTime - startTime >= TIMEOUT_MS) {
        displayString(4, "Movement timeout");
    }
}

// ? Function: Draw hangman parts with error checks
void draw() {
    if (numWrong > MAX_WRONG) return;

    switch (numWrong) {
        case 1: moveGantry(50, 0); break;
        case 2: moveGantry(0, 50); break;
        case 3: moveGantry(-30, 30); break;
        case 4: moveGantry(30, 30); break;
        case 5: moveGantry(-30, -30); break;
        case 6: moveGantry(30, -30); gameOver = true; break;
    }
    wait1Msec(1000);
}

// ? Function: Check for duplicate guesses
bool isDuplicateGuess(char guess) {
    for (int i = 0; i < numWrong; i++) {
        if (incorrectGuesses[i] == guess) {
            return true;
        }
    }
    for (int i = 0; i < numRight; i++) {
        if (correctGuesses[i] == guess) {
            return true;
        }
    }
    return false;
}

// ? Function: Check if the guess is correct
bool checkGuess(char guess) {
    bool correct = false;

    for (int i = 0; i < MAX_WORD_SIZE && word[i] != '\0'; i++) {
        if (word[i] == guess) {
            guessedWord[i] = guess;
            correctGuesses[numRight++] = guess;
            correct = true;
        }
    }

    return correct;
}

// ? Function: Handle user input with RobotC buttons
char enterGuess() {
    displayString(5, "Press a button to guess");

    char guess = 'A' + random(26);  // Simulated random letter

    while (!nNxtButtonPressed) {
        wait1Msec(10);
    }

    displayString(6, "You guessed: %c", guess);

    // Wait for button release
    while (nNxtButtonPressed) {
        wait1Msec(10);
    }

    return guess;
}

// ? Main Execution Loop
task main() {
    calibrate(abort);

    if (abort) {
        displayString(6, "Calibration aborted");
        wait1Msec(2000);
        return;
    }

    while (!gameOver && !abort) {
        char guess = enterGuess();

        if (isDuplicateGuess(guess)) {
            displayString(7, "Duplicate guess: %c", guess);
            continue;
        }

        if (checkGuess(guess)) {
            displayString(7, "Correct guess!");
        } else {
            incorrectGuesses[numWrong++] = guess;
            draw();
        }

        // Check win/loss conditions
        if (numWrong >= MAX_WRONG) {
            gameOver = true;
            win = false;
        } else if (numRight >= strlen(word)) {
            gameOver = true;
            win = true;
        }
    }

    // End game screen
    if (win) {
        displayString(8, "You win!");
    } else {
        displayString(8, "Game over!");
    }

    wait1Msec(2000);
}
