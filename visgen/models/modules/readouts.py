import os
from typing import Union
import torch,torch.fft as fft,torch.nn as nn,torch.nn.functional as F
from visgen.utils.hdc import FPECodebook,RNSCodebook
class CosineCircConv(nn.Module):
	def __init__(self,readout:str,hidden_dim:int,attributes:list,attributes_out_dim:list,attributes_var:list,exppath,dist:str='normal',trainable_codebook=False,pos_delta=.1,moduli=None,load_codebooks:bool=False,verbose=0,device:Union[torch.device,str]=torch.device('cpu'),generator=None,base_phasor=None,keys=None):
		assert len(attributes)==len(attributes_out_dim);super(CosineCircConv,self).__init__();self.hidden_dim=hidden_dim;self.attributes=attributes;self.attributes_out_dim=attributes_out_dim;self.attributes_var=attributes_var;self.dist=dist;self.exppath=exppath;self.readout=readout;self.pos_delta=pos_delta;self.moduli=moduli;self.device=torch.device(device)if isinstance(device,str)else device;self.generator=generator;n_keys=sum(len(m)for m in moduli)if readout=='RNS'else len(attributes)
		if keys is None:self.keys=nn.ParameterList([nn.Parameter(torch.randn(self.hidden_dim).to(device),requires_grad=False)for _ in range(n_keys)])
		else:self.keys=keys
		self.codebooks=[];os.makedirs(self.exppath,exist_ok=True)
		if load_codebooks:
			if self.readout=='RNS':
				for(att,modulus)in zip(attributes,moduli):
					cb_path=os.path.join(self.exppath,f"{att}_{modulus}_codebook.pt")
					if os.path.isfile(cb_path):self.codebooks.append(torch.load(cb_path))
			else:
				for att in attributes:
					cb_path=os.path.join(self.exppath,f"{att}_codebook.pt")
					if os.path.isfile(cb_path):self.codebooks.append(torch.load(cb_path))
		else:
			if self.readout=='HRR':0
			elif self.readout=='FPE':
				for(i,att)in enumerate(attributes):
					if verbose:print(f"Generating codebook for attribute {att}...")
					codebook=FPECodebook(in_features=hidden_dim,out_features=attributes_out_dim[i],var=attributes_var[i],dist=dist,position_delta=pos_delta,device=device,generator=generator,base_phasor=base_phasor);cb=codebook.generate();self.base_phasor=codebook.base_phasor;cb=F.normalize(cb)
					if trainable_codebook:cb=nn.Parameter(cb,requires_grad=trainable_codebook)
					self.codebooks.append(cb)
			elif self.readout=='RNS':
				for(i,att)in enumerate(attributes):
					codebooks=RNSCodebook(in_features=hidden_dim,moduli=moduli[i],device=device).generate()
					for j in range(len(codebooks)):cb=codebooks[j];cb=F.normalize(cb);codebooks[j]=nn.Parameter(cb,requires_grad=trainable_codebook)
					self.codebooks.extend(codebooks)
			if self.readout=='RNS':
				for(i,att)in enumerate(attributes):
					for(j,mod)in enumerate(moduli[i]):cb_path=os.path.join(self.exppath,f"{att}_mod_{mod}_codebook.pt");idx=sum(len(m)for m in self.moduli[:i])+j;torch.save(self.codebooks[idx],cb_path)
			else:
				for(i,att)in enumerate(attributes):cb_path=os.path.join(self.exppath,f"{att}_codebook.pt");torch.save(self.codebooks[i],cb_path)
	@staticmethod
	def _get_logit(x,key,codebook):unbound=fft.ifft(fft.fft(key)*fft.fft(x)).real;return F.cosine_similarity(unbound.unsqueeze(1).float(),codebook.unsqueeze(0).repeat(unbound.shape[0],1,1).float(),dim=-1)
	def forward(self,input,label=None):
		'Key-Value readout.\n\n        Unbind the input vectors using the original key vector.\n        Compute the cosine similarity between the result of the unbinding\n        operation and the codebook associated with that attribute, to get\n        a distribution over the N possible values of that attribute.\n        This operation is repeated for each attribute; the resulting logits\n        are stacked on a new dimension and returned.\n\n        Args:\n            input (_type_): _description_\n            label (_type_, optional): _description_. Defaults to None.\n\n        Returns:\n            _type_: _description_\n        ';x=F.normalize(input,dim=-1);out=[]
		if self.readout=='RNS':
			for i in range(len(self.attributes)):
				mod_logits=[]
				for j in range(len(self.moduli[i])):idx=sum(len(m)for m in self.moduli[:i])+j;logit=self._get_logit(x,self.keys[idx],self.codebooks[idx]);mod_logits.append(logit)
				out.append(mod_logits)
		else:
			for i in range(len(self.attributes)):logit=self._get_logit(x,self.keys[i],self.codebooks[i]);out.append(logit)
		return out
	def reinitialize(self,**kwargs):
		for(key,value)in kwargs.items():
			if hasattr(self,key):setattr(self,key,value)
		resample=kwargs.get('resample',True);base_phasor=None if resample else self.base_phasor;keys=None if resample else self.keys;self.__init__(readout=self.readout,hidden_dim=self.hidden_dim,attributes=self.attributes,attributes_out_dim=self.attributes_out_dim,attributes_var=self.attributes_var,dist=self.dist,exppath=self.exppath,pos_delta=self.pos_delta,device=self.device,generator=self.generator,base_phasor=base_phasor,keys=keys)