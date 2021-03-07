# Overt and covert speech classification using CNNs

...

## Setup

The requirements are minimal and straightforward for the time being, and the modules only require `numpy`, `matplotlib`, and `pandas`, preferably on Python 3.5+...

## Preliminary Processing

At this stage, the `emg_modules.py` module provides basic functionality to traverse the data directory structure, read the binary signal arrays, and parse the metadata files that accompany them (with heavy use of regular expressions). The code can be run simply using `python emg.py` and will produce a dataframe containing metadata for the corresponding signals in a separate array.

To break it down:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import emg

data_dir = 'data_sample'
df, data = emg.parse_data(data_dir, verbose=False)

# this dataframe contains the metadata
print(df)

# this is a list containing all the HD-EMG signals, with 66 channels each
print('Read signals:', len(data))

# select a signal and a channel to read
idx, ch = 0, 0

# load signal and corresponding metadata
sig, meta = data[idx], df.iloc[idx]

# construct time-axis using the parsed sampling rate
t = np.linspace(0, sig.shape[1]/meta.Fs, sig.shape[1])

# visualize
fig, ax = plt.subplots()
ax.plot(t, sig[ch, :])
fig.show()
```
