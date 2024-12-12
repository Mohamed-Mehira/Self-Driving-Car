import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motors():
    def __init__(self, enaR, in1R, in2R, enaL, in1L, in2L):
        self.enaR = enaR
        self.in1R = in1R
        self.in2R = in2R
        self.enaL = enaL
        self.in1L = in1L
        self.in2L = in2L

        GPIO.setup(self.enaR, GPIO.OUT)
        GPIO.setup(self.in1R, GPIO.OUT)
        GPIO.setup(self.in2R, GPIO.OUT)
        GPIO.setup(self.enaL, GPIO.OUT)
        GPIO.setup(self.in1L, GPIO.OUT)
        GPIO.setup(self.in2L, GPIO.OUT)

        self.pwmA = GPIO.PWM(self.enaR, 100)
        self.pwmA.start(0)
        self.pwmB = GPIO.PWM(self.enaL, 100)
        self.pwmB.start(0)

    def move(self, speed=0.5, turn=0, t=0):
        speed *= 100
        turn *= 100

        leftSpeed = speed + turn
        rightSpeed = speed - turn

        if leftSpeed > 100:
            leftSpeed = 100
        elif leftSpeed < -100:
            leftSpeed = -100

        if rightSpeed > 100:
            rightSpeed = 100
        elif rightSpeed < -100:
            rightSpeed = -100
 
        self.pwmA.ChangeDutyCycle(abs(leftSpeed))
        self.pwmB.ChangeDutyCycle(abs(rightSpeed))
 
        if leftSpeed > 0:
            GPIO.output(self.in1R, GPIO.HIGH)
            GPIO.output(self.in2R, GPIO.LOW)
        else:
            GPIO.output(self.in1R, GPIO.LOW)
            GPIO.output(self.in2R, GPIO.HIGH)
 
        if rightSpeed > 0:
            GPIO.output(self.in1L, GPIO.HIGH)
            GPIO.output(self.in2L, GPIO.LOW)
        else:
            GPIO.output(self.in1L, GPIO.LOW)
            GPIO.output(self.in2L, GPIO.HIGH)

        sleep(t)

    def stop(self, t=0):
        self.pwmA.ChangeDutyCycle(0);
        self.pwmB.ChangeDutyCycle(0);
        sleep(t)
 

def main():
    motors.move(0.6, 0, 2)
    motors.stop(2)
    motors.move(-0.5, 0.2, 2)
    motors.stop(2)
 

if __name__ == '__main__':
    motors = Motors(2, 3, 4, 17, 22, 27)
    main()