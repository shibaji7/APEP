from pynasonde.vipir.riq.parsers.read_riq import (
    VIPIR_VERSION_MAP,
    RiqDataset, Pulset,
)
from pynasonde.vipir.riq.datatypes.sct import SctType
from pynasonde.vipir.riq.datatypes.pct import Ionogram, PctType
import numpy as np
from loguru import logger

files = [
    "RADIO_2024099114603.RIQ",
    "WI937_2024099000003.RIQ",
]

file, vipir_config = files[0], VIPIR_VERSION_MAP.configs[1]

riq = RiqDataset(file)
riq.unicode = "latin-1"
riq.sct, riq.pulses, riq.pulsets = SctType(), [], []
with open(file, mode="rb") as f:
    # Read SCT (System Configuration Table) data
    riq.sct.read_sct_from_file_pointer(f, riq.unicode)
    # Read SCT.Station Data
    riq.sct.station.read_station_from_file_pointer(f, riq.unicode)
    # Read SCT.Timing Data
    riq.sct.timing.read_timing_from_file_pointer(f, riq.unicode)
    # Read SCT.Frequency Data
    riq.sct.frequency.read_frequency_from_file_pointer(f, riq.unicode)
    # Read SCT.Reciever Data
    riq.sct.receiver.read_reciever_from_file_pointer(f, riq.unicode)
    # Read SCT.Exciter Data
    riq.sct.exciter.read_exciter_from_file_pointer(f, riq.unicode)
    # Read SCT.Monitor Data
    riq.sct.monitor.read_monitor_from_file_pointer(f, riq.unicode)
    # Fix all SCT strings
    riq.sct.fix_SCT_strings()

    riq.sct.station.rx_count = 2  # Manually set rx_count for testing

    # Load all PRI, PCT, and pulse data
    for j in range(1, riq.sct.timing.pri_count + 1):
        # Create and load PCT (Pulse Configuration Table) data
        pct = PctType().read_pct_from_file_pointer(
            f, riq.sct, vipir_config, riq.unicode
        )
        riq.pulses.append(pct)
        print(pct.record_id)
        if j==1: break
    print(pct)
    # If tune_type is 1, group pulses into sets of pulse_count
    if riq.sct.frequency.tune_type == 1:
        pulset = Pulset()
        for j, pulse in zip(range(1, riq.sct.timing.pri_count + 1), riq.pulses):
            # Add PCT data to the current pulse set
            pulset.append(pulse)
            # Group pulses into sets of pulse_count
            if np.mod(j, riq.sct.frequency.pulse_count) == 0:
                riq.pulsets.append(pulset)
                pulset = Pulset()
    # If tune_type is >=4, group pulses based on special frequency and pulse_count
    elif riq.sct.frequency.tune_type >= 4:
        riq.swap_pulsets = []
        riq.swap_frequency = riq.sct.frequency.base_table[1]
        pulset = Pulset()
        for j, pulse in zip(range(1, riq.sct.timing.pri_count + 1), riq.pulses):
            if pulse.frequency == riq.swap_frequency:
                # Add PRI and PCT data to the current pulse set
                riq.swap_pulsets.append(pulse)
            else:
                # Add PRI and PCT data to the current pulse set
                pulset.append(pulse)
            # Group pulses into sets of pulse_count
            if np.mod(j, riq.sct.frequency.pulse_count * 2) == 0:
                riq.pulsets.append(pulset)
                pulset = Pulset()
        logger.info(
            f"Swap Frequency: {riq.swap_frequency}, Number of swap_pulsets: {len(riq.swap_pulsets)}"
        )
    else:
        raise NotImplementedError(
            f"tune_type {riq.sct.frequency.tune_type} not implemented"
        )
    # Log the number of pulses and pulse sets
    logger.info(
        f"Number of pulses: {riq.sct.timing.pri_count}, and PRI Count: {riq.sct.timing.pri_count}, Pset Count:{riq.sct.frequency.pulse_count}, Pulset: {len(riq.pulsets)}"
    )

riq.sct.dump_sct(file.replace(".RIQ", "_SCT_DUMP.txt"))

# RiqDataset.create_from_file(
#     files[0],
#     unicode="latin-1",
#     vipir_config=VIPIR_VERSION_MAP.configs[1]
# )