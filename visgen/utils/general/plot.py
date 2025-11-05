from itertools import combinations
import matplotlib.patches as patches,matplotlib.pyplot as plt,numpy as np,torch,torch.nn.functional as F,torchvision.utils as vutils
def plot_reconstructed(gt,pred,N):tiles=torch.cat((gt[:N,None,:,:,:],pred[:N,None,:,:,:]),dim=1).flatten(end_dim=1);return vutils.make_grid(tiles,nrow=2,pad_value=.8)
def plot_box(path,img,rect_bnd=None):
	fig,ax=plt.subplots();plt.axis('off');ax.imshow(img)
	if rect_bnd:rect=patches.Rectangle((rect_bnd[0],rect_bnd[1]),rect_bnd[2],rect_bnd[3],linewidth=1,edgecolor='r',facecolor='none');ax.add_patch(rect)
	plt.text(10,10,f"img: {img.shape}",color='white');plt.savefig(path);plt.close();return fig
def plot_codebooks_similarity(mat,attributes):
	if isinstance(mat[0],list):mat=mat[0]
	plt.xticks(np.arange(0,11,1.))
	for(m,att)in zip(mat,attributes):s=F.cosine_similarity(m.unsqueeze(0),m.unsqueeze(1),dim=-1);plt.plot(s[0].cpu().detach(),marker='v',label=att)
	plt.legend(loc='best');intra_cb=plt.gcf();plt.close()
	if len(mat)>1:
		combs=list(combinations(range(len(mat)),2));fig,axes=plt.subplots(1,len(combs))
		if len(combs)==1:axes=[axes]
		for(col,(i,j))in enumerate(combs):similarity=F.cosine_similarity(mat[i].unsqueeze(0),mat[j].unsqueeze(1),dim=-1).cpu().detach();img=axes[col].imshow(similarity,cmap='viridis',interpolation='nearest',vmin=0,vmax=1);axes[col].set_title(f"Similarity {attributes[i]}-{attributes[j]}")
		fig.colorbar(img)
	inter_cb=plt.gcf();plt.close();return intra_cb,inter_cb