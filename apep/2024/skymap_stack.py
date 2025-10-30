from pathlib import Path
import sys

sys.path.extend(
    [
        str(Path(__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[2]),
    ]
)

from typing import Iterable, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
import datetime as dt

from skymap import create_skymaps_panels

size = 15
import scienceplots
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Tahoma",
    "DejaVu Sans",
    "Lucida Grande",
    "Verdana",
]
mpl.rcParams.update(
    {"xtick.labelsize": size, "ytick.labelsize": size, "font.size": size}
)


def select_sky_files(
    folder: Path,
    count: int = 16,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
) -> List[Path]:
    """Return evenly sampled SKY files between the provided time bounds."""
    paths = sorted(Path(folder).glob("*.SKY"))
    if not paths:
        raise FileNotFoundError(f"No SKY files found in {folder}")

    total = len(paths)
    print(f"{folder}: discovered {total} SKY files")

    def _parse_datetime(path: Path) -> dt.datetime:
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"Unable to parse timestamp from {path}")
        token = parts[1]
        year = int(token[:4])
        doy = int(token[4:7])
        hh = int(token[7:9])
        mm = int(token[9:11])
        ss = int(token[11:13])
        base = dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1)
        return base.replace(hour=hh, minute=mm, second=ss)

    timestamps = [_parse_datetime(p) for p in paths]

    if start:
        start = start.replace(tzinfo=None)
    if end:
        end = end.replace(tzinfo=None)

    filtered = [
        p
        for p, ts in zip(paths, timestamps)
        if (start is None or ts >= start) and (end is None or ts <= end)
    ]
    trimmed = filtered if filtered else paths

    if not trimmed:
        trimmed = paths

    if count >= len(trimmed):
        return trimmed

    step = max(1, len(trimmed) // count)
    selection = trimmed[::step][:count]
    if len(selection) < count:
        selection = selection + trimmed[-(count - len(selection)) :]
    return selection[:count]


def build_panels(
    files: Iterable[Path],
    title: str,
    output: Path,
    nrows: int = 4,
    ncols: int = 4,
):
    file_list = [str(Path(f)) for f in files]
    create_skymaps_panels(
        file_list,
        nrows=nrows,
        ncols=ncols,
        fig_title=title,
        fname=str(output),
        wl="X", 
        file_ext="_150km_alleof.nc",
        data_folder="data/2024/mask/",
    )


def main():
    out_dir = Path("figures/2024")
    out_dir.mkdir(parents=True, exist_ok=True)

    kr835_folder = Path(
        "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/SKYWAVE_DPS4D_2024_04_08"
    )
    ws833_folder = Path(
        "/tmp/chakras4/Crucial X9/APEP/AFRL_Digisondes/Digisonde Files/WSMR_DPS4D_2024_04_08"
    )

    kr835_files = select_sky_files(kr835_folder, 16, dt.datetime(2024, 4, 8, 17), dt.datetime(2024, 4, 8, 20, 10))
    ws833_files = select_sky_files(ws833_folder, 16, dt.datetime(2024, 4, 8, 17), dt.datetime(2024, 4, 8, 20, 10))

    build_panels(
        kr835_files,
        title="08 Apr 2024 / KR835",
        output=out_dir / "sky_stack_KR835.png",
    )
    build_panels(
        ws833_files,
        title="08 Apr 2024  / WS833",
        output=out_dir / "sky_stack_WS833.png",
    )
    import os
    os.system(f"cp {out_dir}/sky_stack_WS833.png  manuscript_figures/Figure08.png")
    os.system(f"cp {out_dir}/sky_stack_KR835.png  manuscript_figures/Figure09.png")


if __name__ == "__main__":
    main()
