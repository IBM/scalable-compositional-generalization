from typing import Any,OrderedDict,Tuple
import torch,torch.nn as nn,torch.nn.functional as F
from torch import Tensor
from torchvision.models.densenet import _DenseBlock,_DenseLayer,_Transition
from.base import BaseModel
model_urls={'densenet121':'https://download.pytorch.org/models/densenet121-a639ec97.pth','densenet161':'https://download.pytorch.org/models/densenet161-8d451a50.pth','densenet169':'https://download.pytorch.org/models/densenet169-b2777c0a.pth','densenet201':'https://download.pytorch.org/models/densenet201-c1103571.pth'}
class ISODenseLayer(_DenseLayer):
	def __init__(self,num_input_features:int,bn_size:int,growth_rate:int,drop_rate:float,relu_parameter:float=.5)->None:
		super().__init__(num_input_features,growth_rate,bn_size,drop_rate)
		if relu_parameter is None:relu_parameter=.5
		self.actv_fn1=nn.PReLU(num_parameters=num_input_features,init=relu_parameter);self.actv_fn2=nn.PReLU(num_parameters=bn_size*growth_rate,init=relu_parameter)
class ISODenseBlock(nn.ModuleDict):
	def __init__(self,num_layers:int,num_input_features:int,bn_size:int,growth_rate:int,drop_rate:float)->None:
		super(ISODenseBlock,self).__init__()
		for i in range(num_layers):layer=ISODenseLayer(num_input_features=num_input_features+i*growth_rate,growth_rate=growth_rate,bn_size=bn_size,drop_rate=drop_rate);self.add_module('denselayer%d'%(i+1),layer)
	def forward(self,init_features:Tensor)->Tensor:
		features=[init_features]
		for(name,layer)in self.items():new_features=layer(features);features.append(new_features)
		return torch.cat(features,1)
class DenseNet(BaseModel):
	'Densenet-BC model class, based on\n    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.\n\n    Args:\n        growth_rate (int) - how many filters to add each layer (`k` in paper)\n        block_config (list of 4 ints) - how many layers in each pooling block\n        num_init_features (int) - the number of filters to learn in the first convolution layer\n        bn_size (int) - multiplicative factor for number of bottle neck layers\n          (i.e. bn_size * k features in the bottleneck layer)\n        drop_rate (float) - dropout rate after each dense layer\n        num_classes (int) - number of classification classes\n'
	def __init__(self,block:str='default',growth_rate:int=32,block_config:Tuple[int,int,int,int]=(6,12,24,16),in_channels:int=3,num_init_features:int=64,bn_size:int=4,drop_rate:float=0,num_classes:int=1000,preprocessing=None,attribute_sizes=None,*kargs:Any,**kwargs:Any)->None:
		super(DenseNet,self).__init__(*kargs,**kwargs);self.preprocessing=preprocessing;self.attribute_sizes=attribute_sizes;self.features=nn.Sequential(OrderedDict([('conv0',nn.Conv2d(in_channels,num_init_features,kernel_size=7,stride=2,padding=3,bias=False)),('norm0',nn.BatchNorm2d(num_init_features)),('relu0',nn.ReLU(inplace=True)),('pool0',nn.MaxPool2d(kernel_size=3,stride=2,padding=1))]));num_features=num_init_features
		for(i,num_layers)in enumerate(block_config):
			if block=='default':blk=_DenseBlock(num_layers=num_layers,num_input_features=num_features,bn_size=bn_size,growth_rate=growth_rate,drop_rate=drop_rate)
			elif block=='iso':blk=ISODenseBlock(num_layers=num_layers,num_input_features=num_features,bn_size=bn_size,growth_rate=growth_rate,drop_rate=drop_rate)
			self.features.add_module('denseblock%d'%(i+1),blk);num_features=num_features+num_layers*growth_rate
			if i!=len(block_config)-1:trans=_Transition(num_input_features=num_features,num_output_features=num_features//2);self.features.add_module('transition%d'%(i+1),trans);num_features=num_features//2
		self.features.add_module('norm5',nn.BatchNorm2d(num_features));self.classifier=nn.Linear(num_features,num_classes)
		for m in self.modules():
			if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
			elif isinstance(m,nn.BatchNorm2d):nn.init.constant_(m.weight,1);nn.init.constant_(m.bias,0)
			elif isinstance(m,nn.Linear):nn.init.constant_(m.bias,0)
	def load_pretrained(self,url:str,progress:bool)->None:pretrained_state=torch.hub.load_state_dict_from_url(url,progress=progress);model_state=self.state_dict();pretrained_state={k:v for(k,v)in pretrained_state.items()if k in self.state_dict()and v.size()==self.state_dict()[k].size()};self.state_dict().update(pretrained_state);self.load_state_dict(model_state)
	def forward(self,x:Tensor)->Tensor:
		if self.preprocessing is not None:
			with torch.no_grad():x=self.preprocessing(x)
		features=self.features(x);out=F.relu(features,inplace=True);out=F.adaptive_avg_pool2d(out,(1,1));out=torch.flatten(out,1);out=self.classifier(out)
		if self.objective=='classification':
			logits=[];j=0
			for n in self.attribute_sizes:logits.append(out[:,j:j+n]);j+=n
			x=logits
		return x
class DenseNet121(DenseNet):
	def __init__(self,num_classes:int,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(growth_rate=32,block_config=(6,12,24,16),num_init_features=64,num_classes=num_classes,**kwargs)
		if pretrained:self.load_pretrained(model_urls['densenet121'],progress=progress)
class DenseNet161(DenseNet):
	def __init__(self,num_classes:int,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(growth_rate=48,block_config=(6,12,36,24),num_init_features=96,num_classes=num_classes,**kwargs)
		if pretrained:self.load_pretrained(model_urls['densenet161'],progress=progress)
class DenseNet169(DenseNet):
	def __init__(self,num_classes:int,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(growth_rate=32,block_config=(6,12,32,32),num_init_features=64,num_classes=num_classes,**kwargs)
		if pretrained:self.load_pretrained(model_urls['densenet169'],progress=progress)
class DenseNet201(DenseNet):
	def __init__(self,num_classes:int,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(growth_rate=32,block_config=(6,12,48,32),num_init_features=64,num_classes=num_classes,**kwargs)
		if pretrained:self.load_pretrained(model_urls['densenet201'],progress=progress)