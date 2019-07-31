import scipy.io as sio



    mat_contents = sio.loadmat('/Users/leifer/workspace/PredictionCode/AML18_moving/BrainScanner20170421_103508_MS/PointsStats2.mat')

import numpy as np
#x,y,z coordinates of each point (new index each frame)(
rawPoints=np.array(np.squeeze(mat_contents['pointStats2']['rawPoints']))

#corresponding index
trackIdx=np.squeeze(np.array(mat_contents['pointStats2']['trackIdx']))


#Get the heatmap values
mat_contents = sio.loadmat('/Users/leifer/workspace/PredictionCode/AML18_moving/BrainScanner20170421_103508_MS/heatData.mat')
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

GroupNeurons=True

if (GroupNeurons==False):
    neuron=1
    print('Calculating a single neuron')
    H, xedges, yedges = np.histogram2d(xPosNN[neuron],yPosNN[neuron], normed=False, bins=(xedges,yedges))
    PDF, xedges, yedges = np.histogram2d(xPosNN[neuron],yPosNN[neuron], normed=True, bins=(xedges,yedges))
    #Histogram WEighted Sum
    Hws, xedges, yedges = np.histogram2d(xPosNN[neuron],yPosNN[neuron], weights=rPhotoCorr[neuron], bins=(xedges,yedges))
    numrows, numcols = Hws.shape
else:
    print('Aggregating all neurons together')


    xPosAllNN = np.squeeze(np.reshape(xPosNN, [-1, 1]))
    yPosAllNN = np.squeeze(np.reshape(yPosNN, [-1, 1]))
    zPosAllNN = np.squeeze(np.reshape(zPosNN, [-1, 1]))


    # MeanNormalized
    rPhotoCorrNorm = np.divide(rPhotoCorr, np.nanmean(rPhotoCorr, axis=1, keepdims=True))





    if True:
        print('Expect to be one:')
        print(np.nanmean(rPhotoCorrNorm[1, :]))
        print('Expect to be not quite one')
        print(np.nanmean(rPhotoCorrNorm[:, 400]))

    rPhotoCorrAllNorm = np.squeeze(np.reshape(rPhotoCorrNorm, [-1, 1]))



    H, xedges, yedges = np.histogram2d(xPosAllNN, yPosAllNN, normed=False, bins=(xedges, yedges))
    PDF, xedges, yedges = np.histogram2d(xPosAllNN, yPosAllNN, normed=True, bins=(xedges, yedges))
    # Histogram WEighted Sum
    Hws, xedges, yedges = np.histogram2d(xPosAllNN, yPosAllNN, weights=rPhotoCorrAllNorm,
                                         bins=(xedges, yedges))



import matplotlib
import matplotlib.pyplot as plt





verbose=False
if verbose:
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


# LUT
LUT = mean = np.divide(Hws, H, out=np.zeros_like(H), where=(H != 0))

plt.figure()
plt.imshow(LUT.T, interpolation='None', origin='high', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.title('Loopk Up table, mean of all neurons, z-scored.. validated')
plt.colorbar()
plt.show()

#### Now let's explore using the LUT as something to invert to correct for motion artifact

plt.figure()
plt.imshow(gPhotoCorr, aspect='auto')
plt.title('GFP photocorrected')
plt.colorbar()
plt.show()


# MeanNormalized
gPhotoCorrNorm = np.divide(gPhotoCorr, np.nanmean(gPhotoCorr, axis=1, keepdims=True))



#For the final plotting we wont to show as f-F0/f0
gPhotoCorr0=np.nanpercentile(gPhotoCorr, 20, axis=1,keepdims=True)


plt.figure()
plt.imshow( np.divide(gPhotoCorr-gPhotoCorr0,gPhotoCorr0), aspect='auto', vmin=-1, vmax=2)
plt.title('GFP photocorrected (F-F0)/F0')
plt.colorbar()
plt.show()



### Do the actual inversion

#get everything into a one dimensional array to match position,
# (we don't care about time right now)
gPhotoCorrAllNorm=np.squeeze(np.reshape(gPhotoCorrNorm, [-1, 1]))


#Setup look up interpolation based lookup function function
xcenters = xedges[:-1]+np.median(np.diff(xedges))/2
ycenters = yedges[:-1]+np.median(np.diff(yedges))/2

from scipy import interpolate
N=len(xcenters)
pts = np.array(np.meshgrid(xcenters,ycenters)).T.reshape((N*N,2))
lookUpLinear = interpolate.LinearNDInterpolator(pts,LUT.ravel())

verbose=True
if verbose:
    print('Confirm that the look up worked..')
    plt.figure()
    plt.scatter(lookUpLinear(pts), LUT.ravel())
    plt.show()

#Perform the Lookup
rInferredFromPosNorm = lookUpLinear(xPos.ravel(), yPos.ravel())


if verbose:
    plt.figure()
    plt.scatter(rPhotoCorrAllNorm, rInferredFromPosNorm,alpha=0.5, marker='o', s=5, lw=0,)
    plt.xlabel('True Rvalue of every neuorn')
    plt.ylabel('R value that is inferred from LUT based on position ')
    plt.show()




###Divide by the inferred red value

#for Green
exlcudeNans=np.logical_not( np.isnan(rInferredFromPosNorm) )
gPosCorr_weird=gPhotoCorrAllNorm
gPosCorr_weird[exlcudeNans] =np.divide(gPhotoCorrAllNorm[exlcudeNans], rInferredFromPosNorm[exlcudeNans])
gPosCorr_weird=np.reshape(gPosCorr_weird,gPhotoCorr.shape)

#do for red as control
rPosCorr_weird=rPhotoCorrAllNorm
rPosCorr_weird[exlcudeNans] =np.divide(rPhotoCorrAllNorm[exlcudeNans], rInferredFromPosNorm[exlcudeNans])
rPosCorr_weird=np.reshape(rPosCorr_weird,rPhotoCorr.shape)


if verbose:
    for neuron in np.random.randint(len(rPhotoCorr),size=4):
        plt.figure()
        plt.scatter(np.reshape(rInferredFromPosNorm, gPhotoCorr.shape)[neuron], rPhotoCorrNorm[neuron],alpha=0.5, marker='o', s=5, lw=0,)
        plt.xlabel('Position based Look Up table (from RFP)')
        plt.ylabel('RFP only photobleaching corrected and normalized to mean')
        plt.title('How much of RFPs signal is captured by position for neuron'+str(neuron))
        plt.show()

    #Let's cehck for each neuron how well its RFP and GFP signal correlate with the position look up table

    expectedRfromLUT=np.reshape(rInferredFromPosNorm, gPhotoCorr.shape)
    metric=np.zeros(gPhotoCorr.shape[0])
    for neuron in range(gPhotoCorr.shape[0]):
        #excludenans
        elementsToInclude = np.logical_and( np.logical_not( np.isnan(expectedRfromLUT[neuron])),
                                            np.logical_not( np.isnan(rPhotoCorr[neuron])) )

        metric[neuron]=np.corrcoef(expectedRfromLUT[neuron][elementsToInclude], rPhotoCorrNorm[neuron][elementsToInclude] )[0,1]
    plt.figure()
    plt.hist(metric[np.logical_not(np.isnan(metric))],bins='auto')
    plt.title('Correlation Coefficients for each RFP neuron-time-pt with its position based look-up value')
    plt.xlabel('correlation coefficient')
    plt.ylabel('count')
    plt.show()




from prediction import provenance as prov
###
if verbose:
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.imshow(rPhotoCorrNorm,aspect='auto',vmin=0,vmax=2)
    plt.colorbar()
    plt.title('rPhotoCorr normalized')

    plt.subplot(4,1,2)
    plt.imshow(np.reshape(rInferredFromPosNorm, gPhotoCorr.shape), vmin=.9,vmax=1.1, aspect='auto')
    plt.colorbar()
    plt.title('Activity inferred only from position')

    plt.subplot(4,1,3)
    plt.imshow(rPosCorr_weird,aspect='auto',vmin=0,vmax=2)
    plt.colorbar()
    plt.title('rPhotoCorr normalized divided by inferred activity')

    ax=plt.subplot(4, 1, 4)
    plt.imshow(rPhotoCorrNorm-rPosCorr_weird, vmin=-.1,vmax=.1, aspect='auto')
    plt.colorbar()
    plt.title('Difference')
    prov.stamp(ax,0,-.3)
    plt.show()
###








#### TODO:
# Need to z-score and repeat DONE
# Need to reshape such that all neurons are lumped together into one bin DONE
# Then need to try applyking correction to GFP and see how it goes..






#data[Strain]['input'][Recording]['Neurons']['RedRaw']
#data[Strain]['input'][Recording]['Neurons']['GreenRaw']



