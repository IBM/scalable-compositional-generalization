from torch.nn import CrossEntropyLoss
class AttributeCrossEntropyLoss:
	def __init__(self)->None:self.name='loss'
	def __call__(self,logits:list,targets:list,weights:list=None):
		loss=0;attribute_losses=[.0]*len(logits)
		if weights is None:weights=[1.]*len(logits)
		ce=CrossEntropyLoss()
		for i in range(len(logits)):attribute_losses[i]=ce(logits[i],targets[...,i].flatten());loss+=weights[i]*attribute_losses[i]
		return loss,attribute_losses
class ModularAttributeCrossEntropyLoss:
	def __init__(self)->None:self.name='loss'
	def __call__(self,logits:list,targets:list,weights:list=None):
		loss=0;attribute_losses=[];ce=CrossEntropyLoss()
		if weights is None:weights=[1.]*len(logits)
		for(i,attribute_logits)in enumerate(logits):
			moduli=[mod_logits.shape[1]for mod_logits in attribute_logits];attribute_loss=0
			for(mod_logits,modulus)in zip(attribute_logits,moduli):mod_targets=targets[...,i].flatten()%modulus;attribute_loss+=ce(mod_logits,mod_targets)
			attribute_losses.append(attribute_loss);loss+=weights[i]*attribute_loss
		return loss,attribute_losses