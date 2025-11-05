import os,dictdiffer,torch,torch.nn as nn,yaml
from omegaconf import OmegaConf
def compare_cfgs(cfg1,cfg2):
	data1_dict=yaml.load(OmegaConf.to_yaml(cfg1,resolve=True),Loader=yaml.FullLoader);data2_dict=yaml.load(OmegaConf.to_yaml(cfg2,resolve=True),Loader=yaml.FullLoader);data1_dict['logger']['run_id']=data2_dict['logger']['run_id']
	if data1_dict!=data2_dict:
		print('Differences detected:')
		for diff in list(dictdiffer.diff(data1_dict,data2_dict)):print(diff)
		return False
	return True
def flatten_cfg(d):
	queue=list(OmegaConf.to_container(d).values());values=[]
	while len(queue):
		el=queue.pop(0)
		if isinstance(el,dict):queue.extend(list(el.values()))
		else:values.append(el)
	return values
def get_lsf_info():infod={'jobid':os.environ.get('LSB_JOBID')};infod['host']=os.environ.get('LSB_HOSTS');infod['queue']=os.environ.get('LSB_QUEUE');infod['rsrcreq']=os.environ.get('LSB_EFFECTIVE_RSRCREQ');infod['is_gpu_available']=torch.cuda.is_available();return infod
def beautify_results(res):
	lines=['Training Results'];max_key_length=max(len(key)for key in res.keys())+1
	for(key,value)in res.items():formatted_key=f"\t{key}:";value_str=f"{value:.6f}";lines.append(f"{formatted_key.ljust(max_key_length)} {value_str.rjust(10)}")
	return'\n'.join(lines)