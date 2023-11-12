import subprocess
import sys

import numpy as np
import pandas as pd

from assurance_case import (
    n_experiments,
    p_correct_navigation,
    p_correct_pose,
    p_no_collision,
    sample_mission_bbn,
)
from gui import App

bbn = sample_mission_bbn(
    n_experiments, p_correct_navigation, p_no_collision, p_correct_pose
)

app = App(bbn=bbn)
app.mainloop()
