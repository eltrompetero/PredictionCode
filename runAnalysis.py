
# standard modules
import numpy as np
import matplotlib.pylab as plt
import h5py
# custom modules 
import dataHandler as dh
import makePlots as mp
import dimReduction as dr

        
###############################################    
# 
#    run parameters
#
###############################################
typ = 'AML32' # possible values AML32, AML18, AML70
condition = 'moving' # Moving, immobilized, chip
first = True # if true, create new HDF5 file
###############################################    
# 
#    load data into dictionary
#
##############################################
folder = '{}_{}/'.format(typ, condition)
dataLog = '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition)
outLoc = "Analysis/{}_{}_results.hdf5".format(typ, condition)
outLocData = "Analysis/{}_{}.hdf5".format(typ, condition)

# data parameters
dataPars = {'medianWindow':5, # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':5, # sgauss window for angle velocity derivative. must be odd
            'rotate':True, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5,  # gauss window for red and green channel
            'interpolateNans': 5,#interpolate gaps smaller than this of nan values in calcium data
            }

dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
keyList = np.sort(dataSets.keys())

## results dictionary 
resultDict = {}
for kindex, key in enumerate(keyList):
    resultDict[key] = {}
# analysis parameters

pars ={'nCompPCA':20, # no of PCA components
        'PCAtimewarp':True, #timewarp so behaviors are equally represented
        'trainingCut': 0.7, # what fraction of data to use for training 
        'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
        'useRank': 0, # use the rank transformed version of neural data for all analyses
        'useDeconv': 0, # use the deconvolved transformed version of neural data for all analyses
        'nCluster': 10, # use the deconvolved transformed version of neural data for all analyses
        'useClust':False,# use clusters in the fitting procedure.
        'periods': np.arange(0, 300) # relevant periods in seconds for timescale estimate
         }

behaviors = ['AngleVelocity', 'Eigenworm3']

###############################################    
# 
# check which calculations to perform
#
##############################################
createIndicesTest = 1#True 
periodogram = 1
half_period = 1
hierclust = 0
bta = 0
pca = 1#False
kato_pca = 1#False
half_pca = 1
corr = 1
predNeur = 0
svm = 0
lasso = 0
elasticnet = 0
# this requires moving animals
if condition != 'immobilized':
    predNeur = 0
    svm = 1
    lasso = 1
    elasticnet = 1#True
###############################################    
# 
# create training and test set indices
# 
##############################################
if createIndicesTest:
    for kindex, key in enumerate(keyList):
        resultDict[key] = {'Training':{}}
        for label in behaviors:
            train, test = dr.createTrainingTestIndices(dataSets[key], pars, label=label)
            resultDict[key]['Training'][label] = {'Train':train  }
            resultDict[key]['Training'][label]['Test']=test
    print "Done generating trainingsets"

###############################################    
# 
# calculate the periodogram of the neural signals
#
##############################################
if periodogram:
    print 'running periodogram(s)'
    for kindex, key in enumerate(keyList):
        resultDict[key]['Period'] = dr.runPeriodogram(dataSets[key], pars, testset = None)
# for half the sample each
if half_period:
    print 'running periodogram(s)'
    for kindex, key in enumerate(keyList):
        # first four min
        half1 = np.arange(0,1440)
        # after 4:30 min
        half2 = np.arange(1620,dataSets[key]['Neurons']['Activity'].shape[1])
        resultDict[key]['Period1Half'] = dr.runPeriodogram(dataSets[key], pars, testset = half1)
        resultDict[key]['Period2Half'] = dr.runPeriodogram(dataSets[key], pars, testset = half2)


###############################################    
# 
# correlation neurons and behavior
#
##############################################
if corr:
    print 'running Correlation.'
    for kindex, key in enumerate(keyList):
        resultDict[key]['Correlation'] = dr.behaviorCorrelations(dataSets[key], behaviors)
    # first four minute correlation
    half1 = np.arange(0,1440)
    for kindex, key in enumerate(keyList):
        resultDict[key]['CorrelationHalf'] = dr.behaviorCorrelations(dataSets[key], behaviors, subset = half1)

###############################################    
# 
# run svm to predict discrete behaviors
#
##############################################
if svm:
    for kindex, key in enumerate(keyList):
        print 'running SVM.'
        splits = resultDict[key]['Training']
        resultDict[key]['SVM'] = dr.discreteBehaviorPrediction(dataSets[key], pars, splits )

###############################################    
# 
# run PCA and store results
#
##############################################
#%%
if pca:
    print 'running PCA'
    for kindex, key in enumerate(keyList):
        resultDict[key]['PCA'] = dr.runPCANormal(dataSets[key], pars)
        
###############################################    
# 
# run Kato PCA
#
##############################################
#%%
if kato_pca:
    print 'running Kato et. al PCA'
    for kindex, key in enumerate(keyList):
        resultDict[key]['katoPCA'] = dr.runPCANormal(dataSets[key], pars, deriv = True)
        
###############################################    
# 
# run split first-second half PCA
#
##############################################
#%%
if half_pca:
    print 'half-split PCA'
    for kindex, key in enumerate(keyList):
        # run PCA on each half
        # first four min
        half1 = np.arange(0,1440)
        # after 4:30 min
        half2 = np.arange(1620,dataSets[key]['Neurons']['Activity'].shape[1])
        resultDict[key]['PCAHalf1'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset = half1)
        resultDict[key]['PCAHalf2'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset = half2)
#%%
###############################################    
# 
# predict neural dynamics from behavior
#
##############################################
if predNeur:
    for kindex, key in enumerate(keyList):
        print 'predicting neural dynamics from behavior'
        resultDict[key]['RevPred'] = dr.predictNeuralDynamicsfromBehavior(dataSets[key], pars)
    plt.show()
#%%
###############################################    
# 
# use agglomerative clustering to connect similar neurons
#
##############################################
if hierclust:
    for kindex, key in enumerate(keyList):
        print 'running clustering'
        resultDict[key]['clust'] = dr.runHierarchicalClustering(dataSets[key], pars)
#%%
###############################################    
# 
# use behavior triggered averaging to create non-othogonal axes
#
##############################################
if bta:
    for kindex, key in enumerate(keyList):
        print 'running BTA'
        resultDict[key]['BTA'] =dr.runBehaviorTriggeredAverage(dataSets[key], pars)
#%%
###############################################    
# 
# linear regression using LASSO
#
##############################################
if lasso:
    print "Performing LASSO.",
    for kindex, key in enumerate(keyList):
        
        splits = resultDict[key]['Training']
        resultDict[key]['LASSO'] = dr.runLasso(dataSets[key], pars, splits, plot=0, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key],splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['LASSO'][tmpKey].update(tmpDict[tmpKey])
        
        tmpDict = dr.reorganizeLinModel(dataSets[key], resultDict[key], splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['LASSO'][tmpKey]=tmpDict[tmpKey]
    
#%%
###############################################    
# 
# linear regression using elastic Net
#
##############################################
if elasticnet:
    for kindex, key in enumerate(keyList):
        print 'Running Elastic Net',  key
        splits = resultDict[key]['Training']
        resultDict[key]['ElasticNet'] = dr.runElasticNet(dataSets[key], pars,splits, plot=0, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key], splits,pars, fitmethod = 'ElasticNet', behaviors = behaviors, )
        for tmpKey in tmpDict.keys():
            resultDict[key]['ElasticNet'][tmpKey].update(tmpDict[tmpKey])
        
        tmpDict = dr.reorganizeLinModel(dataSets[key], resultDict[key], splits, pars, fitmethod = 'ElasticNet', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['ElasticNet'][tmpKey]=tmpDict[tmpKey]
    

#%%
###############################################    
# 
# save data as HDF5 file
#
##############################################
dh.saveDictToHDF(outLoc, resultDict)
dh.saveDictToHDF(outLocData, dataSets)