from pynasonde.vipir.riq.parsers.read_riq import (
    VIPIR_VERSION_MAP,
    RiqDataset,
)
from pynasonde.vipir.riq.datatypes.sct import SctType

files = [
    "RADIO_2024099114603.RIQ",
    "WI937_2024099000003.RIQ",
]

# file = files[1]

# riq = RiqDataset(file)
# riq.unicode = "latin-1"
# riq.sct, riq.pulses, riq.pulsets = SctType(), [], []
# with open(file, mode="rb") as f:
#     # Read SCT (System Configuration Table) data
#     riq.sct.read_sct_from_file_pointer(f, riq.unicode)
#     # Read SCT.Station Data
#     riq.sct.station.read_station_from_file_pointer(f, riq.unicode)
#     # Read SCT.Timing Data
#     riq.sct.timing.read_timing_from_file_pointer(f, riq.unicode)
#     # Read SCT.Frequency Data
#     riq.sct.frequency.read_frequency_from_file_pointer(f, riq.unicode)
#     # Read SCT.Reciever Data
#     riq.sct.receiver.read_reciever_from_file_pointer(f, riq.unicode)
#     # Read SCT.Exciter Data
#     riq.sct.exciter.read_exciter_from_file_pointer(f, riq.unicode)
#     # Read SCT.Monitor Data
#     riq.sct.monitor.read_monitor_from_file_pointer(f, riq.unicode)
#     # Fix all SCT strings
#     riq.sct.fix_SCT_strings()

# riq.sct.dump_sct(file.replace(".RIQ", "_SCT_DUMP.txt"))

RiqDataset.create_from_file(
    files[0],
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[0]
)