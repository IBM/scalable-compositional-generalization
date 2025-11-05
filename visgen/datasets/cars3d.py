import numpy as np,torchvision.transforms
from visgen.datasets.base import BaseDataset
class Cars3D(BaseDataset):
	def __init__(self,*args,**kwargs):self._attribute_indices={'orientation':0,'elevation':1,'type':2};self.transform=torchvision.transforms.ToTensor();super().__init__(*args,**kwargs)
	def __getitem__(self,idx:int):
		image=self._dataset_images[idx];targets=self._dataset_targets[idx]
		if self.transform is not None:image=self.transform(image)
		return image,targets
	def _load_data(self,dataset_path,dataset_subset):
		images,targets=[],[]
		with open(f"{dataset_path}/images.npy",'rb')as f:images=np.load(f)
		with open(f"{dataset_path}/targets.npy",'rb')as f:targets=np.load(f)
		targets=np.expand_dims(targets,axis=1);return images,targets
	def _attribute_to_index(self,attribute):return self._attribute_indices[attribute]