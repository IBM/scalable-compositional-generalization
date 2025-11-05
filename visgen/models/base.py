import os,torch
from visgen.utils.general import plot_box
class BaseModel(torch.nn.Module):
	def __init__(self,attributes=None,objective=None,loss_fn=None,metric_fns:list=None,*args,**kwargs):
		super().__init__();self.attributes=attributes;self.objective=objective;self.loss_fn=loss_fn;self.metric_fns=metric_fns
		if loss_fn and metric_fns:self._logged_metrics=[self.loss_fn.name]+[m.name for m in metric_fns];self._logged_metrics+=[f"attributes/{self.loss_fn.name}_{a}"for a in attributes];self._logged_metrics+=[f"attributes/{m.name}_{a}"for a in attributes for m in metric_fns]
	def next_step(self):0
	def get_logged_metrics(self):return self._logged_metrics
	def _compute_metrics(self,yp,y):
		metrics,att_metric=[],[]
		for metric_fn in self.metric_fns:m,am=metric_fn(yp,y);metrics.append(m);att_metric.append(am)
		return metrics,att_metric
	def _compose_logging_dict(self,loss,attr_loss,metrics,attr_metrics):
		attr_loss={f"attributes/{self.loss_fn.name}_{n}":ls.item()for(n,ls)in zip(self.attributes,attr_loss)};met,attr_met={},{}
		for(i,m)in enumerate(self.metric_fns):
			met[m.name]=metrics[i].item()
			for(att,val)in zip(self.attributes,attr_metrics[i]):attr_met[f"attributes/{m.name}_{att}"]=val.item()
		return{self.loss_fn.name:loss.item()}|met|attr_loss|attr_met
	def train_step(self,x,y,optimizer,amp_scaler=None,**kwargs):
		optimizer.zero_grad()
		if amp_scaler:
			with torch.cuda.amp.autocast():yp=self(x);loss,attr_loss=self.loss_fn(yp,y);amp_scaler.scale(loss).backward();total_grad_norm=torch.nn.utils.clip_grad_norm_(self.parameters(),max_norm=1e3)
			if total_grad_norm.isfinite:amp_scaler.step(optimizer);amp_scaler.update()
		else:yp=self(x);loss,attr_loss=self.loss(yp,y);loss.backward();total_grad_norm=torch.nn.utils.clip_grad_norm_(self.parameters(),max_norm=1e3);optimizer.step()
		metrics,attr_metrics=self._compute_metrics(yp,y);return self._compose_logging_dict(loss,attr_loss,metrics,attr_metrics)
	@torch.no_grad()
	def validation_step(self,x,y=None,**kwargs):yp=self(x);loss,attr_loss=self.loss_fn(yp,y);metrics,attr_metrics=self._compute_metrics(yp,y);return self._compose_logging_dict(loss,attr_loss,metrics,attr_metrics)
	@torch.no_grad()
	def plot_debug(self,x,path,**kwargs):self.train();original=plot_box(img=x[22].permute(1,2,0).cpu().numpy(),path=os.path.join(path,'original.png'));x_train=self.preprocessing(x);train_augm=plot_box(img=x_train[22].permute(1,2,0).cpu().numpy(),path=os.path.join(path,'train_augm.png'));self.eval();x_test=self.preprocessing(x);test_augm=plot_box(img=x_test[22].permute(1,2,0).cpu().numpy(),path=os.path.join(path,'test_augm.png'));return[original,train_augm,test_augm],['original','train_augm','test_augm']