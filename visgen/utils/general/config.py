'Custom resolvers for OmegaConf'
from visgen.datasets import IRAVEN
def num_pos(constellation_code:str)->int:
	if constellation_code=='all':return max(pos[-1]for pos in IRAVEN.POSITION_MAP.values())
	constellation_name=IRAVEN.CONSTELLATION_CODE_TO_NAME[constellation_code];return len(IRAVEN.POSITION_MAP[constellation_name])