import numpy as np,torch.nn as nn
from.funct import get_activation
class FC_vec(nn.Module):
	def __init__(self,in_dim=2,out_dim=1,l_hidden=None,activation=None,out_activation=None,**kwargs):
		super(FC_vec,self).__init__();self.in_dim=in_dim;self.out_dim=out_dim;l_neurons=l_hidden+[out_dim];activation=activation+[out_activation];l_layer=[];prev_dim=in_dim
		for[n_hidden,act]in zip(l_neurons,activation):
			l_layer.append(nn.Linear(prev_dim,n_hidden));act_fn=get_activation(act)
			if act_fn is not None:l_layer.append(act_fn)
			prev_dim=n_hidden
		self.net=nn.Sequential(*l_layer)
	def forward(self,x):return self.net(x)
class FC_image(nn.Module):
	def __init__(self,in_dim=64**2,out_dim=2,l_hidden=None,activation=None,out_activation=None,in_channels=1,out_channels=1):
		super(FC_image,self).__init__();self.in_dim=in_dim*in_channels;self.out_dim=out_dim*out_channels;self.out_channels=out_channels;l_neurons=l_hidden+[self.out_dim];activation=activation+[out_activation];l_layer=[];prev_dim=self.in_dim
		for(i_layer,[n_hidden,act])in enumerate(zip(l_neurons,activation)):
			l_layer.append(nn.Linear(prev_dim,n_hidden));act_fn=get_activation(act)
			if act_fn is not None:l_layer.append(act_fn)
			prev_dim=n_hidden
		self.net=nn.Sequential(*l_layer)
	def forward(self,x):
		if len(x.size())==4:x=x.view(x.size(0),-1);out=self.net(x)
		else:out_dim=int(np.sqrt(self.out_dim/self.out_channels));out=self.net(x);out=out.reshape(-1,self.out_channels,out_dim,out_dim)
		return out