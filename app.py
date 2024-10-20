import numpy as np
import cv2 as cv
import time
from random import randint

class MyPerson:
    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
    
    def getRGB(self):
        return (self.R, self.G, self.B)
    
    def getId(self):
        return self.i
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn
    
    def setDone(self):
        self.done = True
    
    def timedOut(self):
        return self.done
    
    def crossingLine(self, line_y):
        if len(self.tracks) >= 2:
            if self.tracks[-1][1] < line_y and self.tracks[-2][1] >= line_y:  # crossed down (entry)
                return 'entry'  # Changed to 'entry' for entering
            elif self.tracks[-1][1] > line_y and self.tracks[-2][1] <= line_y:  # crossed up (exit)
                return 'exit'  # Changed to 'exit' for exiting
        return None
    
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True

# Initialize counters
cnt_entry = 0
cnt_exit = 0

# Video capture
cap = cv.VideoCapture("test3.avi")

# Frame dimensions
h, w = 480, 640
frameArea = h * w
areaTH = frameArea / 250

# Entry/exit line
line_y = int(5 * (h / 7))  # Adjust this value for the line's height
line_color = (255, 0, 0)  # Red color for the line

# Background subtractor
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True)

# Structural elements for morphological filters
kernelOp = np.ones((3, 3), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

# Variables
font = cv.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1
min_area = 500  # Minimum contour area to consider as a valid detection

try:
    log = open('log.txt', "w")
except:
    print("Cannot open log file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Age every person one frame
    for person in persons:
        person.age_one()

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Binarization and morphological operations
    ret, imBin = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
    mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernelCl)

    contours0, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Modify detection logic
    for cnt in contours0:
        area = cv.contourArea(cnt)
        if area > min_area:  # Only consider contours above the minimum area
            M = cv.moments(cnt)
            if M['m00'] == 0:  # Prevent division by zero
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            new_person = True
            for person in persons:
                # Check proximity and update if close enough
                if abs(cx - person.getX()) < 20 and abs(cy - person.getY()) < 20:  # Proximity threshold
                    new_person = False
                    person.updateCoords(cx, cy)
                    crossing = person.crossingLine(line_y)
                    if crossing == 'entry':
                        cnt_entry += 1
                        log.write(f"ID: {person.getId()} is entering the college at {time.strftime('%c')}\n")
                    elif crossing == 'exit':
                        cnt_exit += 1
                        log.write(f"ID: {person.getId()} is exiting the college at {time.strftime('%c')}\n")
                    break

            # If it's a new person, add them to the tracking list
            if new_person:
                new_person_instance = MyPerson(pid, cx, cy, max_p_age)
                persons.append(new_person_instance)
                pid += 1

            # Draw detection
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv.rectangle(frame, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 255, 0), 2)

    # Draw the detection line
    cv.line(frame, (0, line_y), (w, line_y), line_color, 2)

    # Display counts
    cv.putText(frame, f'Exiting the college: {cnt_entry}', (10, 40), font, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, f'Entering the college: {cnt_exit}', (10, 80), font, 0.5, (255, 255, 255), 2, cv.LINE_AA)

    # Show the frame
    cv.imshow('Frame', frame)

    if cv.waitKey(30) & 0xff == 27:  # Break on 'ESC'
        break

log.flush()
log.close()
cap.release()
cv.destroyAllWindows()
