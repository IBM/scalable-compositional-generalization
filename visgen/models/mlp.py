import torch
from visgen.utils.general import plot_box
from.base import BaseModel
__all__=['MLP']
class MLP(BaseModel):
	def __init__(self,net,preprocessing,attribute_sizes,*args,**kwargs):super().__init__(*args,**kwargs);self.net=net;self.preprocessing=preprocessing;self.attribute_sizes=attribute_sizes
	def forward(self,x):
		with torch.no_grad():x=self.preprocessing(x)
		x=x.reshape(x.shape[0],-1);x=self.net(x)
		if self.objective=='classification':
			logits=[];j=0
			for n in self.attribute_sizes:logits.append(x[:,j:j+n]);j+=n
			x=logits
		return x