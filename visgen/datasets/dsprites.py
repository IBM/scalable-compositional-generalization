import numpy as np
from torchvision import transforms
from visgen.datasets.base import BaseDataset
class DSprites(BaseDataset):
	def __init__(self,*args,**kwargs):self._attribute_indices={'color':0,'shape':1,'scale':2,'orientation':3,'x-position':4,'y-position':5};self.transform=transforms.ToTensor();super().__init__(*args,**kwargs)
	def __getitem__(self,idx:int,normalize:bool=False):
		image=self._dataset_images[idx];targets=self._dataset_targets[idx]
		if normalize:targets=targets/(np.array(self._factor_sizes)-1)
		if self.transform is not None:image=self.transform(image)
		return image,targets
	def _load_data(self,dataset_path,dataset_subset):dataset=np.load(dataset_path);dataset_images,dataset_targets=dataset['imgs'],dataset['latents_classes'];dataset_targets=np.expand_dims(dataset_targets,axis=1);dataset_images*=255;return dataset_images,dataset_targets
	def _attribute_to_index(self,attribute):return self._attribute_indices[attribute]