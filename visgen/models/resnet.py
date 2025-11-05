from typing import Optional
import torch,torch.hub,torch.nn as nn,torch.nn.functional as F
from visgen.utils.general import plot_box
from.base import BaseModel
__all__=['ResNet18','ResNet34','ResNet50','ResNet18Decoder']
model_urls={'resnet18':'https://download.pytorch.org/models/resnet18-f37072fd.pth','resnet34':'https://download.pytorch.org/models/resnet34-b627a593.pth','resnet50':'https://download.pytorch.org/models/resnet50-0676ba61.pth','resnet101':'https://download.pytorch.org/models/resnet101-cd907fc2.pth','resnet152':'https://download.pytorch.org/models/resnet152-f82ba261.pth'}
def conv3x3(in_planes,out_planes,stride=1,groups=1,dilation=1):'3x3 convolution with padding';return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=dilation,groups=groups,bias=False,dilation=dilation)
def conv1x1(in_planes,out_planes,stride=1):'1x1 convolution';return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)
class Bottleneck(nn.Module):
	expansion:int=4
	def __init__(self,inplanes:int,planes:int,activation,stride:int=1,downsample=None,groups:int=1,base_width:int=64,dilation:int=1,norm_layer=None,skip_init=False)->None:
		super().__init__()
		if norm_layer is None:norm_layer=nn.BatchNorm2d
		width=int(planes*(base_width/64.))*groups;self.conv1=conv1x1(inplanes,width);self.bn1=norm_layer(width);self.conv2=conv3x3(width,width,stride,groups,dilation);self.bn2=norm_layer(width);self.conv3=conv1x1(width,planes*self.expansion);self.bn3=norm_layer(planes*self.expansion);self.activation=activation;self.downsample=downsample;self.stride=stride
	def forward(self,x):
		identity=x;out=self.conv1(x);out=self.bn1(out);out=self.activation(out);out=self.conv2(out);out=self.bn2(out);out=self.activation(out);out=self.conv3(out);out=self.bn3(out)
		if self.downsample is not None:identity=self.downsample(x)
		out+=identity;out=self.activation(out);return out
class BasicBlock(nn.Module):
	expansion=1
	def __init__(self,inplanes,planes,activation,stride=1,downsample=None,groups=1,base_width=64,dilation=1,norm_layer=None,skip_init=False):
		super(BasicBlock,self).__init__()
		if norm_layer is None:norm_layer=nn.BatchNorm2d
		if groups!=1 or base_width!=64:raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation>1:raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
		self.conv1=conv3x3(inplanes,planes,stride);self.bn1=norm_layer(planes);self.activation=activation;self.conv2=conv3x3(planes,planes);self.bn2=norm_layer(planes);self.downsample=downsample;self.stride=stride
	def forward(self,x):
		identity=x;out=self.conv1(x);out=self.bn1(out);out=self.activation(out);out=self.conv2(out);out=self.bn2(out)
		if self.downsample is not None:identity=self.downsample(x)
		out+=identity;out=self.activation(out);return out
class SReLU(nn.Module):
	def __init__(self,nc,relu_parameter=-1):'initialises shifted ReLU\n        Args:\n            nc: number of image channels of input/output torch.Tensor\n            relu_parameter: initial value of the offset\n';super(SReLU,self).__init__();self.srelu_bias=nn.Parameter(torch.Tensor(1,nc,1,1));self.srelu_relu=nn.ReLU(inplace=True);nn.init.constant_(self.srelu_bias,relu_parameter)
	def forward(self,x):'applies sReLU\n        Args:\n            x: input torch.Tensor\n        Returns:\n            output torch.Tensor after application of sReLU\n';return self.srelu_relu(x-self.srelu_bias)+self.srelu_bias
class BasicISOBlock(BasicBlock):
	def __init__(self,inplanes:int,planes:int,activation,stride:int=1,downsample:Optional[nn.Module]=None,groups=1,base_width=64,dilation=1,norm_layer:Optional[nn.Module]=None,relu_parameter:float=-1,skip_init:bool=False)->None:
		'Coincides with BasicBlock (configured as in wideresnet), but with shifted relu.\n        Initializes a block of a resnet model\n        Args:\n            inplanes: number of channels of input torch.Tensor\n            planes: number of channels of output torch.Tensor and intermediate torch.Tensor after first convolution\n            stride: stride to apply in first convolution\n            downsample: None or 1x1 convolution in case direct skip connection is not possible (channel difference)\n            norm_layer: normalization layer like BatchNorm\n            relu_parameter: initialisation of sReLU parameter\n            skip_init: whether skipInit is used\n';super().__init__(inplanes,planes,activation,stride,downsample,groups,base_width,dilation,norm_layer)
		if relu_parameter is None:relu_parameter=-1
		self.relu1=SReLU(inplanes,relu_parameter=relu_parameter);self.relu2=SReLU(planes,relu_parameter=relu_parameter);self.alpha=torch.nn.parameter.Parameter(data=torch.tensor(.0),requires_grad=True);self.skip_init=skip_init
	def forward(self,x:torch.Tensor)->torch.Tensor:
		identity=x;out=self.conv1(x);out=self.bn1(out);out=self.activation(out);out=self.conv2(out);out=self.bn2(out)
		if self.downsample is not None:identity=self.downsample(x)
		out=self.alpha*out+identity if self.skip_init else out+identity;out+=identity;out=self.activation(out);return out
class AdjustedISOBlock(BasicBlock):
	def __init__(self,inplanes:int,planes:int,activation,stride:int=1,downsample:Optional[nn.Module]=None,groups=1,base_width=64,dilation=1,norm_layer:Optional[nn.Module]=None,relu_parameter:float=.5,skip_init:bool=False)->None:
		'Coincides with BasicBlock (configured as in wideresnet), but with parametric relu.\n        Initializes a block of a resnet model\n        Args:\n            inplanes: number of channels of input tensor\n            planes: number of channels of output tensor and intermediate tensor after first convolution\n            stride: stride to apply in first convolution\n            downsample: None or 1x1 convolution in case direct skip connection is not possible (channel difference)\n            norm_layer: normalization layer like BatchNorm\n            relu_parameter: initialisation of pReLU parameter\n            skip_init: whether skipInit is used\n';super().__init__(inplanes,planes,activation,stride,downsample,groups,base_width,dilation,norm_layer)
		if relu_parameter is None:relu_parameter=.5
		self.relu1=nn.PReLU(num_parameters=inplanes,init=relu_parameter);self.relu2=nn.PReLU(num_parameters=planes,init=relu_parameter);self.alpha=torch.nn.parameter.Parameter(data=torch.tensor(.0),requires_grad=True);self.skip_init=skip_init
	def forward(self,x:torch.Tensor)->torch.Tensor:
		identity=x;out=self.conv1(x);out=self.bn1(out);out=self.relu2(out);out=self.conv2(out);out=self.bn2(out)
		if self.downsample is not None:identity=self.downsample(x)
		out=self.alpha*out+identity if self.skip_init else out+identity;out=self.relu2(out);return out
class ISOBottleneck(Bottleneck):
	def __init__(self,inplanes:int,planes:int,activation,stride:int=1,downsample=None,groups:int=1,base_width:int=64,dilation:int=1,norm_layer=None,relu_parameter=None,skip_init=False)->None:
		super().__init__(inplanes,planes,activation,stride,downsample,groups,base_width,dilation,norm_layer)
		if relu_parameter is None:relu_parameter=.5
		self.relu1=nn.PReLU(num_parameters=planes,init=relu_parameter);self.relu2=nn.PReLU(num_parameters=planes*self.expansion,init=relu_parameter);self.alpha=torch.nn.parameter.Parameter(data=torch.tensor(.0),requires_grad=True);self.skip_init=skip_init
	def forward(self,x):
		identity=x;out=self.conv1(x);out=self.bn1(out);out=self.relu1(out);out=self.conv2(out);out=self.bn2(out);out=self.relu1(out);out=self.conv3(out);out=self.bn3(out)
		if self.downsample is not None:identity=self.downsample(x)
		if self.skip_init:out=self.alpha*out+identity
		else:out+=identity
		out=self.relu2(out);return out
class ResNet(BaseModel):
	def __init__(self,block,layers,in_channels=1,out_dim=512,zero_init_residual=False,groups=1,width_per_group=64,replace_stride_with_dilation=None,norm_layer=None,pretrained=False,maxpool=1,preprocessing=None,head=None,weight_init='kaiming_normal',attribute_sizes=None,activation=None,skip_init=False,*args,**kwargs):
		super().__init__(*args,**kwargs);self.args,self.kwargs=args,kwargs;self.block=block;self.layers=layers;self.preprocessing=preprocessing;self.skip_init=skip_init;self.pretrained=pretrained;self.width_per_group=width_per_group;self.zero_init_residual=zero_init_residual;self.replace_stride_with_dilation=replace_stride_with_dilation;self.attribute_sizes=attribute_sizes;self.weight_init=weight_init
		if norm_layer is None:norm_layer=nn.BatchNorm2d
		elif norm_layer=='identity':norm_layer=nn.Identity
		if activation is None:activation=nn.ReLU(inplace=True)
		self._norm_layer=norm_layer;self.inplanes=64;self.dilation=1;self.in_channels=in_channels;self.out_dim=out_dim
		if replace_stride_with_dilation is None:replace_stride_with_dilation=[False,False,False]
		if len(replace_stride_with_dilation)!=3:raise ValueError(f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}")
		self.groups=groups;self.base_width=width_per_group
		if maxpool==0:self.conv1=nn.Conv2d(self.in_channels,self.inplanes,kernel_size=7,stride=1,padding=3,bias=False);self.maxpool=nn.Identity()
		else:self.conv1=nn.Conv2d(self.in_channels,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False);self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.bn1=norm_layer(self.inplanes);self.activation=activation;self.layer1=self._make_layer(block,64,layers[0],activation,skip_init=skip_init);self.layer2=self._make_layer(block,128,layers[1],activation,stride=2,dilate=replace_stride_with_dilation[0],skip_init=skip_init);self.layer3=self._make_layer(block,256,layers[2],activation,stride=2,dilate=replace_stride_with_dilation[1],skip_init=skip_init);self.layer4=self._make_layer(block,512,layers[3],activation,stride=2,dilate=replace_stride_with_dilation[2],skip_init=skip_init);self.avgpool=nn.AdaptiveAvgPool2d((1,1));self.Tanh=nn.Tanh()
		if 512*block.expansion!=out_dim:self.fc=nn.Linear(512*block.expansion,out_dim)
		self.head=head
		for m in self.modules():
			if isinstance(m,nn.Conv2d)and self.weight_init=='kaiming_normal':nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
			elif isinstance(m,nn.Conv2d)and self.weight_init=='dirac':nn.init.dirac_(m.weight)
			elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):nn.init.constant_(m.weight,1);nn.init.constant_(m.bias,0)
			elif isinstance(m,nn.Conv2d)and self.weight_init=='kaiming_normal_leaky':nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='leaky_relu',a=.5)
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m,BasicBlock):nn.init.constant_(m.bn2.weight,0)
	def _make_layer(self,block,planes,blocks,activation,stride=1,dilate=False,skip_init=False):
		norm_layer=self._norm_layer;downsample=None;previous_dilation=self.dilation
		if dilate:self.dilation*=stride;stride=1
		if stride!=1 or self.inplanes!=planes*block.expansion:downsample=nn.Sequential(conv1x1(self.inplanes,planes*block.expansion,stride),norm_layer(planes*block.expansion))
		layers=[block(self.inplanes,planes,activation,stride,downsample,self.groups,self.base_width,previous_dilation,norm_layer,skip_init=skip_init)];self.inplanes=planes*block.expansion
		for _ in range(1,blocks):layers.append(block(self.inplanes,planes,activation,groups=self.groups,base_width=self.base_width,dilation=self.dilation,norm_layer=norm_layer,skip_init=skip_init))
		return nn.Sequential(*layers)
	def forward(self,x):
		if self.preprocessing is not None:
			with torch.no_grad():x=self.preprocessing(x)
		x=self.conv1(x);x=self.bn1(x);x=self.activation(x);x=self.maxpool(x);x=self.layer1(x);x=self.layer2(x);x=self.layer3(x);x=self.layer4(x);x=self.avgpool(x);x=torch.flatten(x,1)
		if hasattr(self,'fc')and self.fc:x=self.fc(x)
		if self.head is not None:x=self.head(x)
		if self.objective=='classification':
			logits=[];j=0
			for n in self.attribute_sizes:logits.append(x[:,j:j+n]);j+=n
			x=logits
		return x
class ResNet18(ResNet):
	def __init__(self,pretrained:bool,progress:bool=True,**kwargs):
		super().__init__(block=BasicBlock,layers=[2,2,2,2],pretrained=pretrained,**kwargs)
		if pretrained:pretrained_state=torch.hub.load_state_dict_from_url(model_urls['resnet18'],progress=progress);model_state=self.state_dict();pretrained_state={k:v for(k,v)in pretrained_state.items()if k in self.state_dict()and v.size()==self.state_dict()[k].size()};self.state_dict().update(pretrained_state);self.load_state_dict(model_state)
class ResNet34(ResNet):
	def __init__(self,pretrained:bool,progress:bool=True,**kwargs):
		super().__init__(block=BasicBlock,layers=[3,4,6,3],pretrained=pretrained,**kwargs)
		if pretrained:pretrained_state=torch.hub.load_state_dict_from_url(model_urls['resnet34'],progress=progress);model_state=self.state_dict();pretrained_state={k:v for(k,v)in pretrained_state.items()if k in self.state_dict()and v.size()==self.state_dict()[k].size()};self.state_dict().update(pretrained_state);self.load_state_dict(model_state)
class ResNet50(ResNet):
	def __init__(self,pretrained:bool,progress:bool=True,**kwargs):
		super().__init__(block=Bottleneck,layers=[3,4,6,3],pretrained=pretrained,**kwargs)
		if pretrained:pretrained_state=torch.hub.load_state_dict_from_url(model_urls['resnet50'],progress=progress);model_state=self.state_dict();pretrained_state={k:v for(k,v)in pretrained_state.items()if k in self.state_dict()and v.size()==self.state_dict()[k].size()};self.state_dict().update(pretrained_state);self.load_state_dict(model_state)
class ResNet101(ResNet):
	def __init__(self,pretrained:bool,progress:bool=True,**kwargs):
		super().__init__(block=Bottleneck,layers=[3,4,23,3],pretrained=pretrained,**kwargs)
		if pretrained:pretrained_state=torch.hub.load_state_dict_from_url(model_urls['resnet101'],progress=progress);model_state=self.state_dict();pretrained_state={k:v for(k,v)in pretrained_state.items()if k in self.state_dict()and v.size()==self.state_dict()[k].size()};self.state_dict().update(pretrained_state);self.load_state_dict(model_state)
class ResNet152(ResNet):
	def __init__(self,pretrained:bool,progress:bool=True,**kwargs):
		super().__init__(block=Bottleneck,layers=[3,8,36,3],pretrained=pretrained,**kwargs)
		if pretrained:pretrained_state=torch.hub.load_state_dict_from_url(model_urls['resnet152'],progress=progress);model_state=self.state_dict();pretrained_state={k:v for(k,v)in pretrained_state.items()if k in self.state_dict()and v.size()==self.state_dict()[k].size()};self.state_dict().update(pretrained_state);self.load_state_dict(model_state)
class WideResNet50(ResNet):
	def __init__(self,pretrained:bool,progress:bool=True,**kwargs):super().__init__(block=Bottleneck,layers=[3,4,6,3],pretrained=pretrained,width_per_group=64*2,**kwargs)
class Interpolate(nn.Module):
	'nn.Module wrapper for F.interpolate.'
	def __init__(self,size=None,scale_factor=None):super().__init__();self.size,self.scale_factor=size,scale_factor
	def forward(self,x):return F.interpolate(x,size=self.size,scale_factor=self.scale_factor)
def resize_conv3x3(in_planes,out_planes,scale=1):
	'upsample + 3x3 convolution with padding to avoid checkerboard artifact.'
	if scale==1:return conv3x3(in_planes,out_planes)
	return nn.Sequential(Interpolate(scale_factor=scale),conv3x3(in_planes,out_planes))
def resize_conv1x1(in_planes,out_planes,scale=1):
	'upsample + 1x1 convolution with padding to avoid checkerboard artifact.'
	if scale==1:return conv1x1(in_planes,out_planes)
	return nn.Sequential(Interpolate(scale_factor=scale),conv1x1(in_planes,out_planes))