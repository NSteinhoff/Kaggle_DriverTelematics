__author__ = 'nikosteinhoff'

import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

np.random.seed(123)

driver_trips = np.random.randn(190, 2)
fake_trips = np.random.randn(10, 2) - 1

fig, ax = plt.subplots()

real, = ax.plot(driver_trips[:, 0], driver_trips[:, 1], 'og', label='real', alpha=0.75)
fake, = ax.plot(fake_trips[:, 0], fake_trips[:, 1], 'or', label='fake', alpha=0.6)
ax.legend()
ax.tick_params(
    axis='both',
    reset=True,
    labelbottom='off',
    labelleft='off')
ax.set_title('Real and fake trips')

if len(sys.argv) > 1 and sys.argv[1] == 'noshow':
    print("Not showing plot! Saved to file.")
    plt.savefig('driver_sample.png')
else:
    plt.show()