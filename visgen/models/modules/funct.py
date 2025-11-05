import torch.nn as nn
def get_activation(s_act):
	if s_act=='relu':return nn.ReLU(inplace=True)
	elif s_act=='sigmoid':return nn.Sigmoid()
	elif s_act=='softplus':return nn.Softplus()
	elif s_act=='linear':return
	elif s_act=='tanh':return nn.Tanh()
	elif s_act=='leakyrelu':return nn.LeakyReLU(.2,inplace=True)
	elif s_act=='softmax':return nn.Softmax(dim=1)
	elif s_act=='selu':return nn.SELU()
	elif s_act=='gelu':return nn.GELU()
	elif s_act=='elu':return nn.ELU()
	else:raise ValueError(f"Unexpected activation: {s_act}")