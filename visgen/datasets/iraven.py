import os,numpy as np,torchvision.transforms as transforms
from visgen.datasets.base import BaseDataset
class IRAVEN(BaseDataset):
	CONSTELLATION_CODE_TO_NAME={'C':'center_single','2x2':'distribute_four','3x3':'distribute_nine','LR':'left_center_single_right_center_single','UD':'up_center_single_down_center_single','O-IG':'in_center_single_out_center_single','O-IC':'in_distribute_four_out_center_single'};POSITION_MAP={'center_single':[0],'distribute_four':[1,2,3,4],'distribute_nine':[5,6,7,8,9,10,11,12,13],'left_center_single_right_center_single':[14,15],'up_center_single_down_center_single':[16,17],'in_center_single_out_center_single':[0,9],'in_distribute_four_out_center_single':[0,18,19,20,21]};_CONSTELLATION_NAMES=list(CONSTELLATION_CODE_TO_NAME.values());_ATTRIBUTE_INDICES={'position':0,'angle':1,'color':2,'size':3,'type':4}
	def __init__(self,max_obj=1,*args,**kwargs):'\n        :param constellation: Constellation to be loaded: one of (all|C|2x2|3x3|LR|UD|O-IG|O-IC).\n        :param max_obj: Maximum number of objects. Samples with more objects will be ignored.\n        ';self._max_obj=max_obj;self.transform=transforms.Compose([transforms.ToTensor(),transforms.functional.invert]);super().__init__(*args,**kwargs)
	def __getitem__(self,index):
		image,targets=self._dataset_images[index],self._dataset_targets[index]
		if self.transform is not None:image=self.transform(image.astype('uint8'))
		return image,targets
	def _attribute_to_index(self,attribute):return self._ATTRIBUTE_INDICES[attribute]
	def _transform_position(self,constellation,targets):
		pos_idx=self._attribute_to_index('position')
		for target in targets:
			for obj in target:
				pos=obj[pos_idx]
				if pos!=-1:obj[pos_idx]=self.POSITION_MAP[constellation][pos]
		return targets
	def _filter_samples(self,images,targets,max_obj):
		mask=[]
		for target in targets:
			padding_rows=np.where(target[:,0]==-1)[0]
			if padding_rows.size==0 and len(target)<=max_obj or padding_rows.size!=0 and padding_rows[0]<=max_obj:mask.append(True)
			else:mask.append(False)
		mask=np.array(mask);images,targets=images[mask],targets[mask];targets=np.array([target[:max_obj]for target in targets]);return images,targets
	def _load_constellation(self,dataset_path,constellation,transform_position=False):
		file_path=os.path.join(dataset_path,f"{constellation}.npz");data=np.load(file_path);images,targets=data['images'],data['targets']
		if transform_position:targets=self._transform_position(constellation,targets)
		return images,targets
	def _load_data(self,dataset_path,dataset_subset):
		if dataset_subset=='all':
			dataset_images,dataset_targets=self._load_constellation(dataset_path,self._CONSTELLATION_NAMES[0],transform_position=True)
			for constellation in self._CONSTELLATION_NAMES[1:]:images,targets=self._load_constellation(dataset_path,constellation,transform_position=True);dataset_images=np.vstack([dataset_images,images]);dataset_targets=np.vstack([dataset_targets,targets])
		else:dataset_images,dataset_targets=self._load_constellation(dataset_path,self.CONSTELLATION_CODE_TO_NAME[dataset_subset])
		dataset_images,dataset_targets=self._filter_samples(dataset_images,dataset_targets,self._max_obj);return dataset_images,dataset_targets