import h5py,numpy as np,torchvision.transforms
from visgen.datasets.base import BaseDataset
class Shapes3D(BaseDataset):
	def __init__(self,*args,**kwargs):self._attribute_indices={'floor':0,'wall':1,'object':2,'scale':3,'shape':4,'orientation':5};self.transform=torchvision.transforms.ToTensor();super().__init__(*args,**kwargs)
	def __getitem__(self,idx:int):
		image=self._dataset_images[idx];targets=self._dataset_targets[idx]
		if self.transform is not None:image=self.transform(image)
		return image,targets
	def _load_data(self,dataset_path,dataset_subset):
		dataset=h5py.File(dataset_path,'r');dataset_images,dataset_targets=dataset['images'][...],dataset['labels'][...];num_attributes=dataset_targets.shape[-1];attribute_values=[np.unique(dataset_targets[:,i])for i in range(num_attributes)]
		for i in range(num_attributes):map_={attr_val:j for(j,attr_val)in enumerate(attribute_values[i])};dataset_targets[:,i]=[*map(map_.get,dataset_targets[:,i])]
		dataset_targets=dataset_targets.astype(int);dataset_targets=np.expand_dims(dataset_targets,axis=1);return dataset_images,dataset_targets
	def _attribute_to_index(self,attribute):return self._attribute_indices[attribute]