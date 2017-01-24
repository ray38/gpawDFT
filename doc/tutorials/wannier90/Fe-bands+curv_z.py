import pylab as pl
import numpy as np
from matplotlib.gridspec import GridSpec
tick_labels=[]
tick_locs=[]
tick_labels.append('$\Gamma$')
tick_locs.append(0)
tick_labels.append(' H'.strip())
tick_locs.append(    2.189269)
tick_labels.append(' P'.strip())
tick_locs.append(    4.085232)
tick_labels.append(' N'.strip())
tick_locs.append(    5.179867)
tick_labels.append('$\Gamma$')
tick_locs.append(    6.727914)
tick_labels.append(' H'.strip())
tick_locs.append(    8.917183)
tick_labels.append(' N'.strip())
tick_locs.append(   10.465230)
tick_labels.append('$\Gamma$')
tick_locs.append(   12.013277)
tick_labels.append(' P'.strip())
tick_locs.append(   13.909240)
tick_labels.append(' N'.strip())
tick_locs.append(   15.003875)
fig = pl.figure()
gs = GridSpec(2, 1,hspace=0.00)
axes1 = pl.subplot(gs[0, 0:])
data = np.loadtxt('Fe-bands.dat')
x=data[:,0]
y=data[:,1]-    9.576000
pl.scatter(x,y,color='k',marker='+',s=0.1)
pl.xlim([0,max(x)])
pl.ylim([-0.65,0.65]) # Adjust this range as needed
pl.plot([tick_locs[0],tick_locs[-1]],[0,0],color='black',linestyle='--',linewidth=0.5)
pl.xticks(tick_locs,tick_labels)
for n in range(1,len(tick_locs)):
   pl.plot([tick_locs[n],tick_locs[n]],[pl.ylim()[0],pl.ylim()[1]],color='gray',linestyle='-',linewidth=0.5)
pl.ylabel('Energy$-$E$_F$ [eV]')
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
axes2 = pl.subplot(gs[1, 0:])
data = np.loadtxt('Fe-curv.dat')
x=data[:,0]
y=data[:,3]
pl.plot(x,y,color='k')
pl.xlim([0,max(x)])
pl.ylim([min(y)-0.025*(max(y)-min(y)),max(y)+0.025*(max(y)-min(y))])
pl.xticks(tick_locs,tick_labels)
for n in range(1,len(tick_locs)):
   pl.plot([tick_locs[n],tick_locs[n]],[pl.ylim()[0],pl.ylim()[1]],color='gray',linestyle='-',linewidth=0.5)
pl.ylabel('$-\Omega_z(\mathbf{k})$  [ $\AA^2$ ]')
outfile = 'Fe-bands+curv_z.pdf'
pl.savefig(outfile)
pl.show()
