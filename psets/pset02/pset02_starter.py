#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn import neighbors
from scipy.interpolate import interp1d

# A useful function for calculating the nearest neighbors:
#   neighbors.KNeighborsRegressor(n_neighbors)

# Use linear interpolation to predict new values on the test data
#   interp1d

# Save the results as "results.csv"