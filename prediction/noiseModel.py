### noiseModel.py
# We suspect that the neural dynamics we observe in freely moving GCaMP worms have three components:
# 1) Neural Signals representing locomotion
# 2) Neural signals representing other biological processes (sensory processing, working memory, etc)
# 3) Noise, including from motion artifact
#
# We know for sure that (1) & (3) exist. We can decode locomotion so  we know that (1) must be persent.
# And we know that noise  (3) is present  because we see it in the GFP moving animals.
#
# To bolster our argument that neural signals representing other biological processes are present in our recordings,
# we will attempt to reject the following null hypothesis:
#
# Null hypothesis is that ONLY neural signals representing locomotion AND noise are present. E.g. only (1) and (3) and
# not (2).
#
#
# We will build uop our null model by taking a real freely moving GFP worm that therefore has only noise (3) and we
# will synthetically generate pure locmootory signals (1) and adjust the relative proportion of noise to
# locomotory signals.
#
# We  will study the predicitive performance of our decoder on this model and also the percent variance explaiend by the
# locomotory siganl to try to assess whether the null model fits our experimental observations.


## Find a good GFP recording to use.

dataset = '/Users/leifer/workspace/PredictionCode/AML18_moving/BrainScanner20160506_160928_MS/heatData.mat'

# Import the recording based on the output of Jeff's 3dBrain matlab analysis pipeline
import scipy.io as sio
mat_contents = sio.loadmat(dataset)
rPhotoCorr = mat_contents['rPhotoCorr']
gPhotoCorr = mat_contents['gPhotoCorr']
Ratio2 = mat_contents['Ratio2']

# Import Monika's ICA'd version (with my modification s.t. there is no z-scoring)
# (Note this section is copy pasted from S2.py)
import prediction.dataHandler as dh
import numpy as np
data = {}
for typ in ['AML18']:
    for condition in ['moving']:
        folder = '/Users/leifer/workspace/PredictionCode/{}_{}/'.format(typ, condition)
        dataLog = '/Users/leifer/workspace/PredictionCode/{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition)
        outLoc = "/Users/leifer/workspace/PredictionCode/Analysis/{}_{}_results.hdf5".format(typ, condition)
        outLocData = "/Users/leifer/workspace/PredictionCode/Analysis/{}_{}.hdf5".format(typ, condition)

        try:
            # load multiple datasets
            dataSets = dh.loadDictFromHDF(outLocData)
            keyList = np.sort(dataSets.keys())
            results = dh.loadDictFromHDF(outLoc)
            # store in dictionary by typ and condition
            key = '{}_{}'.format(typ, condition)
            data[key] = {}
            data[key]['dsets'] = keyList
            data[key]['input'] = dataSets
            data[key]['analysis'] = results
        except IOError:
            print typ, condition, 'not found.'
            pass
print 'Done reading data.'




import matplotlib.pyplot as plt
from prediction import provenance as prov


ordIndx=data['AML18_moving']['input']['BrainScanner20160506_160928']['Neurons']['ordering']

plt.figure()
plt.subplot(4, 1, 1)
plt.imshow(rPhotoCorr[ordIndx,:],aspect='auto')
plt.colorbar()
plt.title('rPhotoCorr')

plt.subplot(4,1,2)
plt.imshow(gPhotoCorr[ordIndx,:],aspect='auto')
plt.colorbar()
plt.title('gPhotoCorr')

plt.subplot(4,1,3)
plt.imshow(Ratio2[ordIndx,:],aspect='auto')
plt.colorbar()
plt.title('Ratio2')


ax=plt.subplot(4, 1, 4)
plt.imshow(data['AML18_moving']['input']['BrainScanner20160506_160928']['Neurons']['ActivityFull'],aspect='auto')
plt.colorbar()
prov.stamp(ax,0,-.3)
plt.title('"ActivityFull" (ICAd, plus Andys modified normalization) \n' +dataset)
plt.show()


## Extract neural weights learned by the SLM from a GCaMP Recording

## Shuffle the neural Weights

## Generate synthetic pure locomotory signals based on the GFP recording

## Combine the GFP and synthetic locomotory signals for a given relative strength (lambda)

## Figure out how normalization should occur.  [Has somethign to do with lambda.]

## Measure Decoder Performance (R^2)

## Measure variance explained

#Get the heatmap values

