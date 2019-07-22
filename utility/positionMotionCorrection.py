import scipy.io as sio



mat_contents = sio.loadmat('/Users/leifer/workspace/PredictionCode/AML18_moving/BrainScanner20160506_155051_MS/PointsStats2.mat')

import numpy as np
#x,y,z coordinates of each point (new index each frame)(
rawPoints=np.array(np.squeeze(mat_contents['pointStats2']['rawPoints']))

#corresponding index
trackIdx=np.squeeze(np.array(mat_contents['pointStats2']['trackIdx']))


#Get the heatmap values
mat_contents = sio.loadmat('/Users/leifer/workspace/PredictionCode/AML18_moving/BrainScanner20160506_155051_MS/heatData.mat')
rPhotoCorr=mat_contents['rPhotoCorr']
gPhotoCorr=mat_contents['gPhotoCorr']


X=0
Y=1
Z=2
xPos=np.empty(rPhotoCorr.shape)
yPos=np.empty(rPhotoCorr.shape)
zPos=np.empty(rPhotoCorr.shape)


#assemble positions by neuron so that we can later easily look up their intensity

for neuron in range(rPhotoCorr.shape[0]):
    for frame in range(len(rawPoints)):
        neuronIndx =  np.where( np.squeeze(trackIdx[frame])==neuron)
        try:
            xPos[neuron][frame] = rawPoints[frame][neuronIndx].item(X)
            yPos[neuron][frame] = rawPoints[frame][neuronIndx].item(Y)
            zPos[neuron][frame] = rawPoints[frame][neuronIndx].item(Z)
        except:
            xPos[neuron][frame] = np.nan
            yPos[neuron][frame] = np.nan
            zPos[neuron][frame] = np.nan
            pass


#Need to find all nan's in RFP and then turn the underlying position values to Nan, for histogram purposes
xPosNN=xPos
xPosNN[np.where(np.isnan(rPhotoCorr)) ] = np.nan

yPosNN=yPos
yPosNN[np.where(np.isnan(rPhotoCorr)) ] = np.nan

zPosNN=yPos
zPosNN[np.where(np.isnan(rPhotoCorr)) ] = np.nan




xedges=range(0,600,10)
yedges=range(0,600,10)
neuron=1

H, xedges, yedges = np.histogram2d(xPosNN[neuron],yPosNN[neuron], normed=False, bins=(xedges,yedges))
PDF, xedges, yedges = np.histogram2d(xPosNN[neuron],yPosNN[neuron], normed=True, bins=(xedges,yedges))
#Histogram WEighted Sum
Hws, xedges, yedges = np.histogram2d(xPosNN[neuron],yPosNN[neuron], weights=rPhotoCorr[neuron], bins=(xedges,yedges))
numrows, numcols = Hws.shape


import matplotlib
import matplotlib.pyplot as plt



def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = X[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)


fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(121, title='imshow: square bins')
ax.format_coord=format_coord
plt.imshow(Hws, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar()
ax = fig.add_subplot(122, title='Weighted counts pcolormesh: actual edges',
        aspect='equal')
X, Y = np.meshgrid(xedges, yedges)
ax.pcolormesh(X, Y, Hws)
ax.format_coord=format_coord
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(PDF, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.format_coord=format_coord
plt.title('PDF')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.format_coord=format_coord
plt.title('Histogram (counts)')
plt.colorbar()
plt.show()


mean = np.divide(Hws, H, out=np.zeros_like(H), where=(H != 0))

plt.figure()
plt.imshow(mean, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.format_coord=format_coord
plt.title('Mean value')
plt.colorbar()
plt.show()



#### TODO:
# Need to z-score and repeat
# Need to reshape such that all neurons are lumped together into one bin
# Then need to try applyking correction to GFP and see how it goes..






#data[Strain]['input'][Recording]['Neurons']['RedRaw']
#data[Strain]['input'][Recording]['Neurons']['GreenRaw']



