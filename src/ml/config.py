import numpy as np
import pandas as pd
import seaborn as sns

def config_printing():
    np.set_printoptions(linewidth=500)
    np.set_printoptions(precision=3)
    pd.set_option('display.width', 500)
    pd.set_option('precision', 3)
