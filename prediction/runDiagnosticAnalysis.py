
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

def actuallyRun(typ='AML32', condition = 'moving'):
#    typ  possible values AML32, AML18, AML70, AML175
#    condition possible values moving, immobilized, chip
    
        
    first = True # if 0true, create new HDF5 file
    transient = 0
    save = True
    ###############################################    
    # 
    #    load data into dictionary
    #
    ##############################################
    folder = '/Users/leifer/workspace/PredictionCode/{}_{}/'.format(typ, condition)
    dataLog = '/Users/leifer/workspace/PredictionCode/{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition)
    outLoc = "/Users/leifer/workspace/PredictionCode/Analysis/{}_{}_results.hdf5".format(typ, condition)
    outLocData = "/Users/leifer/workspace/PredictionCode/Analysis/{}_{}.hdf5".format(typ, condition)
    
    # data parameters
    dataPars = {'medianWindow':50, # smooth eigenworms with gauss filter of that size, must be odd
                'gaussWindow':100, # sgauss window for angle velocity derivative. must be odd
                'rotate':False, # rotate Eigenworms using previously calculated rotation matrix
                'windowGCamp': 6,  # gauss window for red and green channel
                'interpolateNans': 6,#interpolate gaps smaller than this of nan values in calcium data
                'perNeuronVarNorm': False, # Normalize variance per neuron?
                                    # True makes all the variances the same
                                    # False rescales the ICA'd signal to the mean and variance of the original GCaMP Signal
                                    #       and then applies a (I-I0) / I0 where I0 is the 20th' percentile value of I per neuron

                }
    
    dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
    keyList = np.sort(dataSets.keys())
    if save:
        dh.saveDictToHDF(outLocData, dataSets)
    
    ## results dictionary 
    resultDict = {}
    for kindex, key in enumerate(keyList):
        resultDict[key] = {}
        resultDict[key]['pars'] = dataPars
    # analysis parameters
    
    pars ={'nCompPCA':20, # no of PCA components
            'PCAtimewarp':False, #timewarp so behaviors are equally represented
            'trainingCut': 0.6, # what fraction of data to use for training 
            'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
            'linReg': 'simple', # ordinary or ransac least squares
            'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
            'useRank': 0, # use the rank transformed version of neural data for all analyses
            'useDeconv': 0, # use the deconvolved transformed version of neural data for all analyses
            'useRaw': 0, # use the deconvolved transformed version of neural data for all analyses
            'nCluster': 10, # use the deconvolved transformed version of neural data for all analyses
            'useClust':False,# use clusters in the fitting procedure.
            'periods': np.arange(0, 300), # relevant periods in seconds for timescale estimate
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
    predPCA = 0
    svm = 0
    lasso = 0
    elasticnet = 0
    lagregression = 0
    # this requires moving animals
    if condition != 'immobilized':
        predNeur = 1
        svm = 0
        lasso = 1
        elasticnet = 1#True
        predPCA = 1
        lagregression = 0
    
    



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
            resultDict[key]['PCARaw'] = dr.runPCANormal(dataSets[key], pars, useRaw=True)
            
            
            #correlate behavior and PCA
            #resultDict[key]['PCACorrelation']=dr.PCACorrelations(dataSets[key],resultDict[key], behaviors, flag = 'PCA', subset = None)
    
    #%%
    ###############################################    
    # 
    # save data as HDF5 file
    #
    ##############################################
    if save:
        dh.saveDictToHDF(outLoc, resultDict)




