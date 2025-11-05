import cv2,numpy as np,torch,torch.nn as nn,torchvision.transforms as T
class ShapeDetector(torch.nn.Module):
	def __init__(self,*args,**kwargs)->None:super().__init__(*args,**kwargs)
	@torch.no_grad()
	def forward(self,x):
		x_cropped=[]
		for im in x:thresh=cv2.threshold(im[0].cpu().numpy()*255,30,255,cv2.THRESH_BINARY)[1];cnts,_=cv2.findContours(thresh.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);list_bbx=[cv2.boundingRect(c)for c in cnts][0];crop=im[0][max(0,list_bbx[1]-10):min(64,list_bbx[1]+list_bbx[2]+10),max(0,list_bbx[0]-10):min(64,list_bbx[0]+list_bbx[3]+10)].unsqueeze(0);left_pad=(64-crop.shape[1])//2;top_pad=(64-crop.shape[2])//2;pad=left_pad,64-left_pad-crop.shape[2],top_pad,64-top_pad-crop.shape[1];x_cropped.append(nn.functional.pad(crop,pad))
		x=torch.stack(x_cropped,dim=0);return x
class EdgeDetector(torch.nn.Module):
	def __init__(self,sigma:float=.33,kernel_size:int=3,*args,**kwargs)->None:super().__init__(*args,**kwargs);self.sigma=sigma;self.kernel_size=kernel_size;a=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).float();self.conv1=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False);self.conv1.weight=nn.Parameter(a.unsqueeze(0).unsqueeze(0));b=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).float();self.conv2=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False);self.conv2.weight=nn.Parameter(b.unsqueeze(0).unsqueeze(0))
	def forward(self,x):
		if x.device!=self.conv1.weight.device:x=x.to(self.conv1.weight.device)
		G_x=self.conv1(x);G_y=self.conv2(x);G=torch.sqrt(torch.pow(G_x,2)+torch.pow(G_y,2));shape=G.shape;G=G.view(G.size(0),-1);G-=G.min(1,keepdim=True)[0];G/=G.max(1,keepdim=True)[0];G=G.view(shape);return G
class Augmentator(nn.Module):
	def __init__(self,train_augm,test_augm,*args,**kwargs)->None:super().__init__(*args,**kwargs);self.train_augm=self._init(train_augm);self.test_augm=self._init(test_augm)
	@torch.no_grad()
	def forward(self,x):augm=self.train_augm if self.training else self.test_augm;return augm(x)
	def _init(self,augms):
		tr=[]
		for augm in augms:
			name=augm.pop('name')
			if name=='padding':tr.append(T.Pad(**augm))
			if name=='rand_crop':tr.append(T.RandomCrop(**augm))
			if name=='gauss_blur':tr.append(T.GaussianBlur(**augm))
			if name=='rand_rotation':tr.append(T.RandomRotation(**augm))
			if name=='resize':tr.append(T.Resize(**augm))
			if name=='jit':tr.append(T.ColorJitter(**augm))
			if name=='center_crop':tr.append(T.CenterCrop(**augm))
			if name=='normalize':tr.append(T.Normalize(**augm))
		return nn.Sequential(*tr)