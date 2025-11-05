from functools import partial
from typing import Any,Callable,List,Optional,Sequence
import torch
from torch import Tensor,nn
from torchvision.models.convnext import CNBlock,CNBlockConfig,LayerNorm2d
from torchvision.ops import Conv2dNormActivation,Permute
from visgen.models.base import BaseModel
model_urls={'convnext_tiny':'https://download.pytorch.org/models/convnext_tiny-983f1562.pth','convnext_small':'https://download.pytorch.org/models/convnext_small-0c510722.pth','convnext_base':'https://download.pytorch.org/models/convnext_base-6075fbad.pth','convnext_large':'https://download.pytorch.org/models/convnext_large-ea097f82.pth'}
class ISOCNBlock(CNBlock):
	def __init__(self,dim,layer_scale:float,stochastic_depth_prob:float,norm_layer:Optional[Callable[...,nn.Module]]=None,relu_parameter:float=.5)->None:
		if norm_layer is None:norm_layer=partial(nn.LayerNorm,eps=1e-06)
		super().__init__(dim,layer_scale,stochastic_depth_prob,norm_layer);self.block=nn.Sequential(nn.Conv2d(dim,dim,kernel_size=7,padding=3,groups=dim,bias=True),Permute([0,2,3,1]),norm_layer(dim),nn.Linear(in_features=dim,out_features=4*dim,bias=True),Permute([0,3,1,2]),nn.PReLU(num_parameters=4*dim,init=relu_parameter),Permute([0,2,3,1]),nn.Linear(in_features=4*dim,out_features=dim,bias=True),Permute([0,3,1,2]))
class ConvNeXt(BaseModel):
	def __init__(self,block_setting:List[CNBlockConfig],stochastic_depth_prob:float=.0,in_channels:int=3,layer_scale:float=1e-06,num_classes:int=1000,block:str='CNBlock',norm_layer:Optional[Callable[...,nn.Module]]=None,preprocessing=None,attribute_sizes=None,*args:Any,**kwargs:Any)->None:
		super().__init__(*args,**kwargs);self.preprocessing=preprocessing;self.attribute_sizes=attribute_sizes
		if not block_setting:raise ValueError('The block_setting should not be empty')
		elif not isinstance(block_setting,Sequence)or not all(isinstance(s,CNBlockConfig)for s in block_setting):raise TypeError('The block_setting should be List[CNBlockConfig]')
		if block=='CNBlock':block=CNBlock
		elif block=='ISOCNBlock':block=ISOCNBlock
		else:raise ValueError(f"Unknown block: {block}")
		if norm_layer is None:norm_layer=partial(LayerNorm2d,eps=1e-06)
		firstconv_output_channels=block_setting[0].input_channels;layers:List[nn.Module]=[Conv2dNormActivation(in_channels,firstconv_output_channels,kernel_size=4,stride=4,padding=0,norm_layer=norm_layer,activation_layer=None,bias=True)];stage_block_id=0;total_stage_blocks=sum(cnf.num_layers for cnf in block_setting)
		for cnf in block_setting:
			stage:List[nn.Module]=[]
			for _ in range(cnf.num_layers):sd_prob=stochastic_depth_prob*stage_block_id/(total_stage_blocks-1.);stage.append(block(cnf.input_channels,layer_scale,sd_prob));stage_block_id+=1
			layers.append(nn.Sequential(*stage))
			if cnf.out_channels is not None:layers.append(nn.Sequential(norm_layer(cnf.input_channels),nn.Conv2d(cnf.input_channels,cnf.out_channels,kernel_size=2,stride=2)))
		self.features=nn.Sequential(*layers);self.avgpool=nn.AdaptiveAvgPool2d(1);lastblock=block_setting[-1];lastconv_output_channels=lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels;self.classifier=nn.Sequential(norm_layer(lastconv_output_channels),nn.Flatten(1),nn.Linear(lastconv_output_channels,num_classes,bias=False))
		for m in self.modules():
			if isinstance(m,(nn.Conv2d,nn.Linear)):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
	def forward(self,x:Tensor)->Tensor:
		if self.preprocessing is not None:
			with torch.no_grad():x=self.preprocessing(x)
		x=self.features(x);x=self.avgpool(x);x=self.classifier(x)
		if self.objective=='classification':
			logits=[];j=0
			for n in self.attribute_sizes:logits.append(x[:,j:j+n]);j+=n
			x=logits
		return x
	def load_pretrained(self,url:str,progress:bool)->None:pretrained_state=torch.hub.load_state_dict_from_url(url,progress=progress);model_state=self.state_dict();pretrained_state={k:v for(k,v)in pretrained_state.items()if k in self.state_dict()and v.size()==self.state_dict()[k].size()};self.state_dict().update(pretrained_state);self.load_state_dict(model_state)
class ConvNeXtTiny(ConvNeXt):
	def __init__(self,num_classes:int,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(block_setting=[CNBlockConfig(96,192,3),CNBlockConfig(192,384,3),CNBlockConfig(384,768,9),CNBlockConfig(768,None,3)],stochastic_depth_prob=kwargs.pop('stochastic_depth_prob',.1),num_classes=num_classes,**kwargs)
		if pretrained:self.load_pretrained(model_urls['convnext_tiny'],progress=progress)
class ConvNeXtSmall(ConvNeXt):
	def __init__(self,num_classes:int,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(block_setting=[CNBlockConfig(96,192,3),CNBlockConfig(192,384,3),CNBlockConfig(384,768,27),CNBlockConfig(768,None,3)],stochastic_depth_prob=kwargs.pop('stochastic_depth_prob',.4),num_classes=num_classes,**kwargs)
		if pretrained:self.load_pretrained(model_urls['convnext_small'],progress=progress)
class ConvNeXtBase(ConvNeXt):
	def __init__(self,num_classes:int,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(block_setting=[CNBlockConfig(128,256,3),CNBlockConfig(256,512,3),CNBlockConfig(512,1024,27),CNBlockConfig(1024,None,3)],stochastic_depth_prob=kwargs.pop('stochastic_depth_prob',.5),num_classes=num_classes,**kwargs)
		if pretrained:self.load_pretrained(model_urls['convnext_base'],progress=progress)
class ConvNeXtLarge(ConvNeXt):
	def __init__(self,num_classes:int,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(block_setting=[CNBlockConfig(192,384,3),CNBlockConfig(384,768,3),CNBlockConfig(768,1536,27),CNBlockConfig(1536,None,3)],stochastic_depth_prob=kwargs.pop('stochastic_depth_prob',.5),num_classes=num_classes,**kwargs)
		if pretrained:self.load_pretrained(model_urls['convnext_large'],progress=progress)