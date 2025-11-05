from.codebooks import FPECodebook,RNSCodebook
def generate_codebook(cfg,device):
	init=cfg.get('init','FPE')
	if init=='FPE':return FPECodebook(in_features=cfg.in_features,out_features=cfg.out_features,var=cfg.var,dist=cfg.get('dist','normal'),position_delta=cfg.get('pos_delta',.4),device=device).generate()
	elif init=='RNS':return RNSCodebook(in_features=cfg.in_features,moduli=cfg.moduli,device=device)
	else:raise ValueError(f"{init} is not a valid initialization scheme")