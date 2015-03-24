__author__ = 'nikosteinhoff'

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

driver = 1
trips = 200

data_folder = '/Users/nikosteinhoff/Data/Kaggle/AxaDriverTelematics/drivers'

total_frames = 0
trip_readings = []
trip_data = []
for trip in range(1, trips + 1):
    path = os.path.join(data_folder, str(driver), str(trip)+'.csv')
    data = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=True)
    trip_data.append(data)
    total_frames += data.shape[0]
    trip_readings.append(data.shape[0])

speed_factor = 20
frames_to_show = int(math.floor(total_frames/speed_factor))

cum_trip_readings = np.array(trip_readings).cumsum().tolist()

# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()
ax.set_title('Driver Telematics')
ax.set_xlabel('x-Position')
ax.set_ylabel('y-Position')

lines = []
for trip in trip_data:
    new_line, = ax.plot([], [], lw=2)
    lines.append(new_line)


# initialization function: plot the background of each frame
def init():
    global lines
    for line in lines:
        line.set_data([], [])
    return lines


global_x_max = 100
global_x_min = -100
global_y_max = 100
global_y_min = -100


# animation function.  This is called sequentially
def animate(i):
    global global_x_min
    global global_x_max
    global global_y_min
    global global_y_max

    global lines
    global trip_data
    global trip_readings
    global cum_trip_readings

    global ax

    cycle = i * speed_factor

    trip = sum([cycle > x-1 for x in cum_trip_readings])
    subtract_rows = [0] + cum_trip_readings
    rows = cycle - subtract_rows[trip]

    x = trip_data[trip][:rows, 0]
    y = trip_data[trip][:rows, 1]
    lines[trip].set_data(x, y)
    lines[trip].set_data(x+10, y+10)

    if rows > 0:
        x_min = x.min()
        x_max = x.max()
        x_margin = (x_max - x_min) / 20
        current_x_max = x_max + x_margin
        if current_x_max > global_x_max:
            global_x_max = current_x_max
        current_x_min = x_min - x_margin
        if current_x_min < global_x_min:
            global_x_min = current_x_min

        y_min = y.min()
        y_max = y.max()
        y_margin = (y_max - y_min) / 20
        current_y_max = y_max + y_margin
        if current_y_max > global_y_max:
            global_y_max = current_y_max
        current_y_min = y_min - y_margin
        if current_y_min < global_y_min:
            global_y_min = current_y_min

    ax.set_xlim(global_x_min, global_x_max)
    ax.set_ylim(global_y_min, global_y_max)
    return lines


# call the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames_to_show, interval=1)

anim.save('driver_plot_seaborn.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# plt.show()

if __name__ == '__main__':
    print("Running as main.")