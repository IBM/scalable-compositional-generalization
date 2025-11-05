from.metrics import Accuracy,ModularMultiAccuracy,MultiAccuracy,RSquared
__all__=['Accuracy','RSquared','MultiAccuracy','ModularMultiAccuracy']
def get_metrics(cfg,*args,version=None,**kwargs):
	metrics=[]
	for m in cfg:
		if m.name=='accuracy':metrics.append(Accuracy())
		elif m.name=='multi_accuracy':metrics.append(MultiAccuracy())
		elif m.name=='modular_multi_accuracy':metrics.append(ModularMultiAccuracy())
		elif m.name=='r2':metrics.append(RSquared())
	return metrics