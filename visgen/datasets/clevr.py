import json,os,numpy as np,torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from visgen.datasets.base import BaseDataset
class CLEVR(BaseDataset):
	def __init__(self,*args,**kwargs):self._attribute_indices={'shape':0,'size':1,'material':2,'color':3};self.transform=transforms.ToTensor();super().__init__(*args,**kwargs)
	def __getitem__(self,idx:int):
		image=self._dataset_images[idx];targets=self._dataset_targets[idx]
		if self.transform is not None:image=self.transform(image)
		return image,targets
	def _load_data(self,dataset_path,dataset_subset):
		images,targets=[],[]
		with open(os.path.join(dataset_path,'images.npy'),'rb')as f:images=np.load(f)
		with open(os.path.join(dataset_path,'targets.npy'),'rb')as f:targets=np.load(f)
		targets=np.expand_dims(targets,axis=1);images=images.transpose((0,2,3,1));return images,targets
	def _attribute_to_index(self,attribute):return self._attribute_indices[attribute]
