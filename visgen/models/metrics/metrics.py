import numpy as np,torch
from visgen.utils.hdc import RNSCodebook
class RSquared:
	def __init__(self,all_targets):self.name='r2';variance_per_factor=((all_targets-all_targets.mean(axis=0,keepdims=True))**2).mean(axis=0);self.variance_per_factor=variance_per_factor
	def __call__(self,predictions,targets):assert predictions.shape==targets.shape;mse_loss_per_factor=torch.mean((predictions-targets)**2,axis=0);return 1-mse_loss_per_factor/self.variance_per_factor
class Accuracy:
	def __init__(self)->None:self.name='acc'
	def __call__(self,predictions,targets):assert predictions.shape==targets.shape;batch_size=targets.shape[0];correct=predictions.eq(targets);acc=correct.float().sum().mul_(1e2/batch_size);return acc,correct
class MultiAccuracy:
	def __init__(self)->None:self.name='acc'
	def __call__(self,predictions,targets):
		attr_acc=[.0]*len(predictions);corrs=[];batch_size=targets.shape[0];accuracy=Accuracy()
		for i in range(len(predictions)):pred=torch.argmax(predictions[i],dim=-1);attr_acc[i],corr=accuracy(pred,targets[...,i].flatten());corrs.append(corr)
		corr_tot=torch.prod(torch.stack(corrs),0);acc_tot=corr_tot.float().sum()/batch_size*100;return acc_tot,attr_acc
class ModularMultiAccuracy:
	def __init__(self)->None:self.name='acc'
	def __call__(self,predictions,targets):
		attr_accs,corrs=[],[];batch_size=targets.shape[0];accuracy=Accuracy()
		for(i,attribute_preds)in enumerate(predictions):
			moduli,remainders=[],[]
			for mod_preds in attribute_preds:moduli.append(mod_preds.shape[1]);remainders.append(mod_preds.argmax(dim=-1).cpu().numpy())
			pred=[RNSCodebook.reconstruct(moduli,rs)for rs in np.transpose(remainders)];pred=torch.Tensor(pred).to(targets.device);attr_acc,corr=accuracy(pred,targets[...,i].flatten());attr_accs.append(attr_acc);corrs.append(corr)
		corr_tot=torch.prod(torch.stack(corrs),0);acc_tot=corr_tot.float().sum()/batch_size*100;return acc_tot,attr_accs