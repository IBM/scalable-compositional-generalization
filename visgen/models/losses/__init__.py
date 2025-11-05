from.losses import AttributeCrossEntropyLoss,ModularAttributeCrossEntropyLoss
__all__=['AttributeCrossEntropyLoss','ModularAttributeCrossEntropyLoss']
def get_loss(cfg,*args,version=None,**kwargs):
	name=cfg.name
	if name=='cross_entropy':loss=AttributeCrossEntropyLoss()
	elif name=='modular_cross_entropy':loss=ModularAttributeCrossEntropyLoss()
	return loss