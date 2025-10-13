from pynasonde.digisonde.parsers.sao import SaoExtractor

sao = SaoExtractor("tmp/MHJ45_20250904(247)000000.SAO", extract_time_from_name=False, extract_stn_from_name=False)
ret = sao.extract()
print(ret.keys())