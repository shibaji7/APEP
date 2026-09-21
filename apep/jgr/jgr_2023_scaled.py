"""
JGR Figure 02: ionosonde response during the 14 Oct 2023 eclipse (5 x 4 time series, 13-21 UT).

Data: Datasets/Data/<STN>_2023_scaled.csv (eclipse day) and <STN>_2023_median.csv (quiet-day
median) for AU930, BC840, AL945, KR835 and WS833. Columns are foF2, h'F2, foE and h'E (the CSVs
hold virtual heights). Event series are gap-filled (<= 90 min) and smoothed; rows are sorted by
maximum obscuration. Plot style is shared with jgr_timeseries_5x4.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from apep.jgr.jgr_timeseries_5x4 import OUT_DIR, make_figure


def create_figure_02(output_dir=OUT_DIR, smooth=True):
    make_figure(2023, output_dir, smooth=smooth)


if __name__ == "__main__":
    create_figure_02()
