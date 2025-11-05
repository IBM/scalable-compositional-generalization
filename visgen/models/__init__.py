import os,torch.nn as nn
from omegaconf import OmegaConf
from visgen.utils.hdc.codebooks import RNSCodebook
from.convnext import ConvNeXtBase,ConvNeXtLarge,ConvNeXtSmall,ConvNeXtTiny
from.densenet import DenseNet121,DenseNet161,DenseNet169,DenseNet201
from.losses import get_loss
from.metrics import get_metrics
from.mlp import MLP
from.modules import FC_image,FC_vec
from.modules.funct import get_activation
from.modules.readouts import CosineCircConv
from.ed import ExpDisentanglement
from.preprocess import Augmentator,EdgeDetector,ShapeDetector
from.resnet import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,WideResNet50
from.ain import SplitResNet18
from.vit import SwinTransformerBase,SwinTransformerTiny,ViT
def get_model(cfg,*args,version=None,**kwargs):name=cfg.model.arch;model=_get_model_instance(name);model=model(**cfg);return model
def _get_model_instance(name):
	try:return{'mlp':get_mlp,'convnext_tiny':get_convnext,'convnext_small':get_convnext,'convnext_base':get_convnext,'convnext_large':get_convnext,'resnet18':get_resnet,'resnet34':get_resnet,'resnet50':get_resnet,'resnet101':get_resnet,'resnet152':get_resnet,'wideresnet':get_resnet,'resnet18_decoder':get_resnet,'densenet121':get_densenet,'densenet161':get_densenet,'densenet169':get_densenet,'densenet201':get_densenet,'vit':get_vit,'swin_base':get_swin,'swin_tiny':get_swin,'ed':get_neuro_sym,'split_resnet':get_split_resnet}[name]
	except ValueError as e:raise f"Unknown model {name}!"from e
def _get_attribute_info(cfg):attributes=cfg['data']['training']['attributes'];targets=cfg['data']['training']['targets'].split('_');used_attributes=[attr for attr in attributes if attr['name']in targets];att_names=[n['name']for n in used_attributes];att_size=[n['out_dim']for n in used_attributes];att_var=[n['var']for n in used_attributes];return att_names,att_size,att_var
def get_mlp(**cfg):
	model_cfg=cfg['model'];l_hidden=model_cfg['l_hidden'];activation=model_cfg['activation'];out_activation=model_cfg['out_activation'];loss=get_loss(cfg['training']['loss']);objective=cfg['training']['objective'];metrics=get_metrics(cfg['training']['metrics']);in_chan=model_cfg['feature_extraction']['in_channels'];height=width=64;in_dim=in_chan*height*width;attribute_names,attribute_sizes,_=_get_attribute_info(cfg)
	if objective=='classification':out_dim=sum(attribute_sizes)
	elif objective=='regression':out_dim=len(attribute_names)
	else:raise NotImplementedError(f"Objective {objective} is not supported.")
	preprocessing=_get_preprocessing(model_cfg.pop('preprocessing'));net=get_net(in_dim=in_dim,out_dim=out_dim,arch='fc_vec',l_hidden=l_hidden,activation=activation,out_activation=out_activation);return MLP(net=net,preprocessing=preprocessing,attributes=attribute_names,attribute_sizes=attribute_sizes,objective=objective,loss_fn=loss,metric_fns=metrics)
def get_convnext(**cfg):
	model_cfg=cfg['model'];loss=get_loss(cfg['training']['loss']);objective=cfg['training']['objective'];metrics=get_metrics(cfg['training']['metrics']);attribute_names,attribute_sizes,_=_get_attribute_info(cfg);preprocessing=_get_preprocessing(model_cfg.pop('preprocessing'))
	if objective=='classification':out_dim=sum(attribute_sizes)
	elif objective=='regression':out_dim=len(attribute_names)
	additional_parameters={'attribute_sizes':attribute_sizes,'preprocessing':preprocessing,'num_classes':out_dim,'attributes':attribute_names,'objective':objective,'loss_fn':loss,'metric_fns':metrics};return get_net(**dict(model_cfg)|additional_parameters)
def get_densenet(**cfg):
	model_cfg=cfg['model'];loss=get_loss(cfg['training']['loss']);objective=cfg['training']['objective'];metrics=get_metrics(cfg['training']['metrics']);attribute_names,attribute_sizes,_=_get_attribute_info(cfg);preprocessing=_get_preprocessing(model_cfg.pop('preprocessing'))
	if objective=='classification':out_dim=sum(attribute_sizes)
	elif objective=='regression':out_dim=len(attribute_names)
	additional_parameters={'attribute_sizes':attribute_sizes,'preprocessing':preprocessing,'num_classes':out_dim,'attributes':attribute_names,'objective':objective,'loss_fn':loss,'metric_fns':metrics};return get_net(**dict(model_cfg)|additional_parameters)
def get_swin(**cfg):
	model_cfg=cfg['model'];loss=get_loss(cfg['training']['loss']);objective=cfg['training']['objective'];metrics=get_metrics(cfg['training']['metrics']);attribute_names,attribute_sizes,_=_get_attribute_info(cfg);preprocessing=_get_preprocessing(model_cfg.pop('preprocessing'))
	if objective=='classification':out_dim=sum(attribute_sizes)
	elif objective=='regression':out_dim=len(attribute_names)
	additional_parameters={'attribute_sizes':attribute_sizes,'preprocessing':preprocessing,'num_classes':out_dim,'attributes':attribute_names,'objective':objective,'loss_fn':loss,'metric_fns':metrics};return get_net(**dict(model_cfg)|additional_parameters)
def get_vit(**cfg):model_cfg=cfg['model'];loss=get_loss(cfg['training']['loss']);objective=cfg['training']['objective'];metrics=get_metrics(cfg['training']['metrics']);in_chan=model_cfg['feature_extraction']['in_channels'];attribute_names,attribute_sizes,_=_get_attribute_info(cfg);preprocessing=_get_preprocessing(model_cfg.pop('preprocessing'));pretrained=model_cfg['pretrained'];return ViT(in_channels=in_chan,preprocessing=preprocessing,attributes=attribute_names,attribute_sizes=attribute_sizes,pretrained=pretrained,objective=objective,loss_fn=loss,metric_fns=metrics)
def get_resnet(**cfg):
	model_cfg=cfg['model'];in_chan=model_cfg['feature_extraction']['in_channels'];emb_dim=model_cfg['emb_dim'];out_activation=model_cfg['out_activation'];loss=get_loss(cfg['training']['loss']);objective=cfg['training']['objective'];metrics=get_metrics(cfg['training']['metrics']);attribute_names,attribute_sizes,_=_get_attribute_info(cfg);preprocessing=_get_preprocessing(model_cfg.pop('preprocessing'))
	if objective=='classification':out_dim=sum(attribute_sizes)
	elif objective=='regression':out_dim=len(attribute_names)
	else:raise NotImplementedError(f"Objective {objective} is not supported.")
	head=get_net(in_dim=emb_dim,out_dim=out_dim,arch='fc_vec',l_hidden=[],activation=[],out_activation=out_activation);return get_net(in_channels=in_chan,out_dim=emb_dim,preprocessing=preprocessing,head=head,attributes=attribute_names,attribute_sizes=attribute_sizes,objective=objective,loss_fn=loss,metric_fns=metrics,**model_cfg)
def get_neuro_sym(**cfg):
	model_cfg=cfg['model'];arch=model_cfg['arch'];model_cfg['feature_extraction']['in_channels'];z_dim=model_cfg['z_dim'];cb_path=os.path.join(model_cfg['path'],'codebooks');att_names,att_size,att_var=_get_attribute_info(cfg);preprocessing=_get_preprocessing(model_cfg.pop('preprocessing'));loss=get_loss(cfg['training']['loss']);objective=cfg['training']['objective'];metrics=get_metrics(cfg['training']['metrics'])
	if arch!='ed':raise ValueError(f"Architecture {arch} not supported")
	f_extractors=[get_net(out_dim=z_dim,**model_cfg['feature_extraction'])for _ in att_names];readouts=[]
	for i in range(len(att_names)):att_dict={'name':[att_names[i]],'siz':[att_size[i]],'var':[att_var[i]],'cbpath':cb_path,'device':cfg['device']};readouts.append(get_net(in_dim=z_dim,out_dim=att_size[i],**dict(model_cfg['readout'])|att_dict))
	return ExpDisentanglement(preprocessing,f_extractors,readouts,objective=objective,attributes=att_names,loss_fn=loss,metric_fns=metrics)
def _get_preprocessing(m):modules=[get_net(**module)for module in m];return nn.Sequential(*modules)if len(modules)else get_net(arch='identity')
def get_net(**kwargs):
	arch=kwargs.pop('arch');in_dim=kwargs.get('in_dim');out_dim=kwargs.get('out_dim');in_channels=kwargs.get('in_channels',1);out_channels=kwargs.get('out_channels',1)
	if arch=='identity':net=nn.Identity()
	elif arch=='fc_vec':net=FC_vec(**kwargs)
	elif arch=='fc_image':l_hidden=kwargs['l_hidden'];activation=kwargs['activation'];out_activation=kwargs['out_activation'];net=FC_image(in_dim=in_dim,out_dim=out_dim,l_hidden=l_hidden,activation=activation,out_activation=out_activation,in_channels=in_channels,out_channels=out_channels)
	elif arch in['resnet18','resnet34','resnet50','resnet101','resnet152','wideresnet','resnet18_decoder']:kwargs.pop('in_dim',None);kwargs.pop('out_channels',None);resnet_classes={'resnet18':ResNet18,'resnet34':ResNet34,'resnet50':ResNet50,'resnet101':ResNet101,'resnet152':ResNet152,'wideresnet':WideResNet50};args=kwargs|{'activation':get_activation(kwargs['activation'])};return resnet_classes[arch](**args)
	elif arch in['densenet121','densenet161','densenet169','densenet201']:kwargs|={'in_channels':kwargs['feature_extraction']['in_channels']};densenet_classes={'densenet121':DenseNet121,'densenet161':DenseNet161,'densenet169':DenseNet169,'densenet201':DenseNet201};return densenet_classes[arch](**kwargs)
	elif arch in['swin_tiny','swin_base']:kwargs|={'in_channels':kwargs['feature_extraction']['in_channels']};swin_classes={'swin_tiny':SwinTransformerTiny,'swin_base':SwinTransformerBase};return swin_classes[arch](**kwargs)
	elif arch in['convnext_tiny','convnext_small','convnext_base','convnext_large']:convnext_classes={'convnext_tiny':ConvNeXtTiny,'convnext_small':ConvNeXtSmall,'convnext_base':ConvNeXtBase,'convnext_large':ConvNeXtLarge};kwargs=kwargs|{'in_channels':kwargs['feature_extraction']['in_channels']};return convnext_classes[arch](**kwargs)
	elif arch=='opencv_detect':net=ShapeDetector()
	elif arch=='opencv_edge':net=EdgeDetector()
	elif arch=='augmentator':net=Augmentator(train_augm=OmegaConf.to_container(kwargs['train']),test_augm=OmegaConf.to_container(kwargs['test']))
	elif arch=='cosine_circ_conv':cb=kwargs['codebook'];moduli=[RNSCodebook.get_moduli(size)for size in kwargs['siz']];pos_delta=kwargs.get('pos_delta',cb['pos_delta']);net=CosineCircConv(readout=cb['init'],hidden_dim=in_dim,attributes=kwargs['name'],attributes_out_dim=kwargs['siz'],attributes_var=kwargs['var'],dist=cb.get('dist','normal'),trainable_codebook=cb['trainable'],pos_delta=pos_delta,moduli=moduli,load_codebooks=cb['load'],exppath=kwargs['cbpath'],generator=kwargs.get('generator',None),device=kwargs.get('device','cpu'))
	elif arch in['split_resnet']:comp_classes={'split_resnet':SplitResNet18};args=kwargs|{'activation':get_activation(kwargs['activation'])};return comp_classes[arch](**args)
	else:raise ValueError(f"{kwargs['arch']} is not a valid network")
	return net
def get_split_resnet(**cfg):
	model_cfg=cfg['model'];in_chan=model_cfg['feature_extraction']['in_channels'];emb_dim=model_cfg['emb_dim'];out_activation=model_cfg['out_activation'];loss=get_loss(cfg['training']['loss']);objective=cfg['training']['objective'];metrics=get_metrics(cfg['training']['metrics']);attribute_names,attribute_sizes,_=_get_attribute_info(cfg);preprocessing=_get_preprocessing(model_cfg.pop('preprocessing'))
	if objective=='classification':out_dim=sum(attribute_sizes)
	elif objective=='regression':out_dim=len(attribute_names)
	else:raise NotImplementedError(f"Objective {objective} is not supported.")
	head=get_net(in_dim=emb_dim,out_dim=out_dim,arch='fc_vec',l_hidden=[],activation=[],out_activation=out_activation);return get_net(in_channels=in_chan,out_dim=emb_dim,preprocessing=preprocessing,head=head,attributes=attribute_names,attribute_sizes=attribute_sizes,objective=objective,loss_fn=loss,metric_fns=metrics,**model_cfg)