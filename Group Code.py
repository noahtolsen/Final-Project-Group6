import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("cc_default_data.csv")

pd.set_option('display.max_columns', None) # Making sure all columns are displayed in the print statement
pd.set_option('display.width', 200) # Making sure the columns aren't printed on two separate lines

print(data.describe().transpose())

