from typing import Any
import torch,torchvision.transforms as T
from torchvision.models import ViT_B_16_Weights,vit_b_16
from torchvision.models.swin_transformer import PatchMergingV2
from torchvision.models.swin_transformer import SwinTransformer as TorchSwinTransformer
from torchvision.models.swin_transformer import SwinTransformerBlockV2
from.base import BaseModel
model_urls={'swintransformer_tiny':'https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth','swintransformer_base':'https://download.pytorch.org/models/swin_v2_b-781e5279.pth'}
class ViT(BaseModel):
	def __init__(self,in_channels,preprocessing,attribute_sizes,pretrained,*args,**kwargs):
		super().__init__(*args,**kwargs);self.model=vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
		if pretrained:
			for param in self.model.parameters():param.requires_grad=False
		self.model.classifier=torch.nn.Linear(768,sum(attribute_sizes));self.in_channels=in_channels;self.preprocessing=preprocessing;self.attribute_sizes=attribute_sizes
	def forward(self,x,label=None):
		with torch.no_grad():x=self.preprocessing(x)
		if self.in_channels==1:x=x.repeat(1,3,1,1)
		x=T.Resize((224,224))(x);x=self.model(x)
		if self.objective=='classification':
			logits=[];j=0
			for n in self.attribute_sizes:logits.append(x[:,j:j+n]);j+=n
			x=logits
		return x
class SwinTransformer(BaseModel):
	def __init__(self,patch_size,embed_dim:int,depths,num_heads,window_size,stochastic_depth_prob:float,in_channels=3,preprocessing=None,attribute_sizes=None,*args,**kwargs):
		super().__init__(*args,**kwargs);self.model=TorchSwinTransformer(patch_size=patch_size,embed_dim=embed_dim,depths=depths,num_heads=num_heads,window_size=window_size,stochastic_depth_prob=stochastic_depth_prob);head_in_features=self.model.head.in_features
		if in_channels==1:self.model.features[0][0]=torch.nn.Conv2d(1,embed_dim,kernel_size=(patch_size[0],patch_size[1]),stride=(patch_size[0],patch_size[1]))
		if attribute_sizes is not None:self.model.head=torch.nn.Linear(head_in_features,sum(attribute_sizes))
		else:self.model.head=torch.nn.Identity()
		self.preprocessing=preprocessing;self.attribute_sizes=attribute_sizes
		for m in self.model.modules():
			if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):torch.nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
	def forward(self,x:torch.Tensor)->torch.Tensor:
		if self.preprocessing is not None:
			with torch.no_grad():x=self.preprocessing(x)
		x=self.model(x)
		if self.objective=='classification':
			logits=[];j=0
			for n in self.attribute_sizes:logits.append(x[:,j:j+n]);j+=n
			x=logits
		return x
	def load_pretrained(self,url:str,progress:bool)->None:pretrained_state=torch.hub.load_state_dict_from_url(url,progress=progress);model_state=self.state_dict();pretrained_state={k:v for(k,v)in pretrained_state.items()if k in self.state_dict()and v.size()==self.state_dict()[k].size()};self.state_dict().update(pretrained_state);self.load_state_dict(model_state)
class SwinTransformerTiny(SwinTransformer):
	def __init__(self,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(patch_size=[4,4],embed_dim=96,depths=[2,2,6,2],num_heads=[3,6,12,24],window_size=[8,8],stochastic_depth_prob=.2,block=SwinTransformerBlockV2,downsample_layer=PatchMergingV2,**kwargs)
		if pretrained:self.load_pretrained(model_urls['swintransformer_tiny'],progress=progress)
class SwinTransformerBase(SwinTransformer):
	def __init__(self,pretrained:bool=False,progress:bool=True,**kwargs:Any)->None:
		super().__init__(patch_size=[4,4],embed_dim=128,depths=[2,2,18,2],num_heads=[4,8,16,32],window_size=[8,8],stochastic_depth_prob=.5,block=SwinTransformerBlockV2,downsample_layer=PatchMergingV2,**kwargs)
		if pretrained:self.load_pretrained(model_urls['swintransformer_base'],progress=progress)