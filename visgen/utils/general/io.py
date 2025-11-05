import json,os,shutil,torch,yaml
from omegaconf import OmegaConf
def save_yaml(path,text):
	'parse string as yaml then dump as a file'
	with open(path,'w')as f:yaml.dump(yaml.safe_load(text),f,default_flow_style=False)
def load_json(path):
	with open(path)as f:d=json.load(f)
	return d
def save_json(path,d):
	with open(path,'w')as fp:json.dump(d,fp,sort_keys=True,indent=4)
def save_yaml_safe(path,cfg):"Safely save omegaconf.\n    If the config already exists in the path, check that it's coherent with the\n    current config (ensure consistence when re-running experiments). Otherwise\n    creates it.\n\n    Args:\n        cfg (OmegaConf): config to be saved\n        path (st): save path\n    ";save_yaml(path,OmegaConf.to_yaml(cfg,resolve=True))
def save_checkpoint(model,epoch,is_best,savepath,best_ams,optimizer=None,filename='checkpoint.pth.tar'):
	path=os.path.join(savepath,filename);path_best=os.path.join(savepath,'model_best.pth.tar');torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict()if optimizer else None,'best_ams':best_ams},path)
	if is_best:shutil.copy(path,path_best)
def load_checkpoint(path,model,optimizer=None,device='cpu'):
	ckpt=torch.load(path,map_location=device);model.load_state_dict(ckpt['model_state_dict'])
	if'optimizer_state_dict'in ckpt and optimizer:optimizer.load_state_dict(ckpt['optimizer_state_dict'])
	epoch=ckpt['epoch'];best_ams=ckpt['best_ams'];return model,best_ams,epoch,optimizer