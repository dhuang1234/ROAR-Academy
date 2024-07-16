from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

# remove the axes
# make the minute and hour hands update
# make a yellow colored GMT hand

# Initialization, define some constant
path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/airplane.bmp'
background = plt.imread(filename)

second_hand_length = 200
second_hand_width = 2
minute_hand_length = 150
minute_hand_width = 6
hour_hand_length = 100
hour_hand_width = 10
center = np.array([256, 256])

def clock_hand_vector(angle, length):
    return np.array([length * np.sin(angle), -length * np.cos(angle)])

def GMT_hand_vector(angle, length):
    return np.array([length*np.cos(angle),length*np.sin(angle)])

# draw an image background, remove the axes
fig, ax = plt.subplots()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

while True:
    plt.imshow(background)

    # First retrieve the time
    now_time = datetime.now() #time right now, not like time.time() which is time from 1970
    gmt = now_time.hour
    hour = now_time.hour
    if hour>12: hour = hour - 12
    minute = now_time.minute
    second = now_time.second

    # Calculate end points of hour, minute, second, with instant updating on all hands
    hour_vector = clock_hand_vector((hour+minute/60+second/3600)*(2*np.pi/12), hour_hand_length)
    minute_vector = clock_hand_vector((minute+second/60)*(2*np.pi/60), minute_hand_length)
    second_vector = clock_hand_vector(second/60*2*np.pi, second_hand_length)
    gmt_vector = GMT_hand_vector((gmt+minute/60+second/3600)*np.pi/12,hour_hand_length)

    plt.arrow(center[0], center[1], hour_vector[0], hour_vector[1], head_length = 3, linewidth = hour_hand_width, color = 'black')
    plt.arrow(center[0], center[1], minute_vector[0], minute_vector[1], linewidth = minute_hand_width, color = 'black')
    plt.arrow(center[0], center[1], second_vector[0], second_vector[1], linewidth = second_hand_width, color = 'red')
    plt.arrow(center[0], center[1], gmt_vector[0], gmt_vector[1], head_length = 3, linewidth = hour_hand_width, color = 'yellow')

    plt.pause(0.1)
    plt.clf() # this wipes the previous plot so that the hands dont get left on their last position, updates the hand and deletes the plot with the previous position
