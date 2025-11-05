import math
from typing import Union
import numpy as np,torch,torch.functional as F
class FPECodebook:
	def __init__(self,in_features:int,out_features:int,var:float,dist='normal',position_delta:Union[float,torch.Tensor]=.4,device:torch.device='cpu',generator:torch.nn.Module=None,base_phasor:torch.Tensor=None)->None:self.in_features=in_features;self.out_features=out_features;self.var=var;self.dist=dist;self.position_delta=position_delta.to(device)if torch.is_tensor(position_delta)else position_delta;self.device=device;self.generator=generator.to(device)if generator is not None else None;self.base_phasor=base_phasor
	def generate(self):
		unit_phasor=torch.zeros(self.in_features);std=math.sqrt(self.var)
		if self.base_phasor is None:
			if self.dist=='uniform':left_half=-math.pi+torch.rand(math.floor((self.in_features-1)/2))*2*math.pi
			elif self.dist=='normal':left_half=torch.randn(math.floor((self.in_features-1)/2))*std*2*math.pi
			elif self.dist=='tanh':left_half=F.tanh(torch.randn(math.floor((self.in_features-1)/2))*std)*2*math.pi
			elif self.dist=='learned':left_half=torch.randn(math.floor((self.in_features-1)/2))*std*2*math.pi;left_half=left_half.to(self.device);left_half=20*self.generator(left_half)
			left_half=left_half.to(self.device);right_half=-torch.flip(left_half,dims=(-1,));unit_phasor=torch.cat((torch.Tensor([0]).to(self.device),left_half))
			if self.in_features%2==0:unit_phasor=torch.cat((unit_phasor,torch.Tensor([0]).to(self.device)))
			base_phasor=torch.cat((unit_phasor,right_half));self.base_phasor=base_phasor
		else:base_phasor=self.base_phasor
		good_fpe_matrix=torch.complex(torch.zeros(self.out_features,self.in_features),torch.zeros(self.out_features,self.in_features))
		for i in range(self.out_features):good_fpe_matrix.imag[i]=base_phasor*(i*self.position_delta+1)
		codebook=torch.real(torch.fft.ifft(torch.exp(good_fpe_matrix)));codebook=codebook/torch.linalg.norm(codebook,axis=-1,keepdims=True);return codebook.to(self.device)
class RNSCodebook:
	def __init__(self,in_features:int,moduli:list,device:torch.device='cpu'):self.in_features=in_features;self.moduli=moduli;self.device=device
	@staticmethod
	def _generate_base_phasor(n,m):k=torch.randint(0,m,(n,));return 2j*torch.pi*k/m
	@staticmethod
	def reconstruct(moduli:list[int],remainders:list[int]):M=np.prod(moduli);m_bar=[int(M/m)for m in moduli];mod_inv=[pow(m_bar,-1,m)for(m_bar,m)in zip(m_bar,moduli)];X=sum(r*mb*inv for(r,mb,inv)in zip(remainders,m_bar,mod_inv));return X%M
	@staticmethod
	def get_moduli(dynamic_range:int):
		if dynamic_range<=6:moduli=[2,3]
		elif dynamic_range<=10:moduli=[2,5]
		elif dynamic_range<=15:moduli=[3,5]
		elif dynamic_range<=28:moduli=[4,7]
		elif dynamic_range<=210:moduli=[5,6,7]
		else:raise RuntimeError('Dynamic range too large for current hard-coded moduli.')
		return moduli
	def generate(self):
		codebooks=[]
		for modulus in self.moduli:
			base_phasor=self._generate_base_phasor(self.in_features,modulus);matrix=torch.complex(torch.zeros(modulus,self.in_features),torch.zeros(modulus,self.in_features))
			for i in range(modulus):matrix[i]=base_phasor*(i+1)
			codebook=torch.real(torch.fft.ifft(torch.exp(matrix)));codebooks.append(codebook.to(self.device))
		return codebooks