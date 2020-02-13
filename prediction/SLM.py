import numpy as np
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
from copy import deepcopy

import dataHandler as dh
import userTracker

def train_tree(X, Y):
    clf = tree.DecisionTreeClassifier(max_depth=1)
    clf = clf.fit(X, np.sign(Y))
    return clf

def R2(P, f, X, Y):
    return 1-np.sum(np.power(Y-f(X,P),2))/np.sum(np.power(Y-np.mean(Y),2))

def split_test(X, test):
    center_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) <= test/2
    train_idx = np.abs((np.arange(X.shape[-1])-0.5*X.shape[-1])/X.shape[-1]) > test/2
    X_train = X.T[train_idx].T
    X_test = X.T[center_idx].T
    return (X_train, X_test)

def optimize_slm(Xfull, Yfull, options = None):
    if options is None:
        options = dict()
    default_options = {
    'derivative_penalty': False,
    'decision_tree': False,
    'best_neuron': False,
    'max_depth': 1,
    'alphas': np.logspace(-1.5, 0.5, 5),
    'l1_ratio': 0.9,
    'sigma': 14,
    'derivative_weight': 10,
    'cv_fraction': 0.4,
    'test_fraction': 0.4
    }
    for k in default_options:
        if k not in options:
            options[k] = default_options[k]

    X, Xtest = split_test(Xfull, options['test_fraction'])
    Y, Ytest = split_test(Yfull, options['test_fraction'])

    if options['best_neuron']:
        f = lambda X, P: P[1]*X + P[0]
    else:
        f = lambda X, P: np.dot(X.T, P[1:]) + P[0]
    reg = lambda P: options['l1_ratio']*np.sum(np.abs(P[1:])) + 0.5*(1-options['l1_ratio'])*np.sum(P[1:]*P[1:])

    error_plain = lambda P, X, Y, alpha: (1./(4*Y.size*np.var(Y)))*np.sum(np.power(Y-f(X,P),2)) + alpha*reg(P)
    if options['derivative_penalty']:
        error = lambda P, X, Y, X_deriv, Y_deriv, alpha: error_plain(P, X, Y, alpha) + options['derivative_weight']*(1/(4*Y.size*np.var(Y_deriv)))*np.sum(np.power(Y_deriv-f(X_deriv, P), 2))
    else:
        error = error_plain

    X_train, X_cv = split_test(X, options['cv_fraction'])
    Y_train, Y_cv = split_test(Y, options['cv_fraction'])
    
    if not options['decision_tree']:
        if not options['best_neuron']:
            if not options['derivative_penalty']:
                best_alpha = 0
                best_R2 = -np.inf
                for alpha in options['alphas']:
                    result = minimize(error, np.zeros(X.shape[0]+1), args=(X_train, Y_train, alpha))
                    r2 = R2(result.x, f, X_cv, Y_cv) 
                    if r2 > best_R2:
                        best_alpha = alpha
                        best_R2 = r2
                
                result = minimize(error, np.zeros(X.shape[0]+1), args=(X, Y, best_alpha))
                P = result.x
                return {'weights' : P[1:],
                        'offset'  : P[0],
                        'params'  : P,
                        'error'   : result.fun,
                        'alpha'   : best_alpha,
                        'R2_train': R2(P, f, X, Y),
                        'R2_test' : R2(P, f, Xtest, Ytest)
                        }
            
            else:
                X_deriv = gaussian_filter(X, sigma = (0, options['sigma']), order=1)
                Y_deriv = gaussian_filter(Y, sigma = options['sigma'], order = 1)
                X_train_deriv = gaussian_filter(X_train, sigma = (0, options['sigma']), order = 1)
                Y_train_deriv = gaussian_filter(Y_train, sigma = options['sigma'], order = 1)

                best_alpha = 0
                best_R2 = -np.inf
                for alpha in options['alphas']:
                    result = minimize(error, np.zeros(X.shape[0]+1), args=(X_train, Y_train, X_train_deriv, Y_train_deriv, alpha))
                    r2 = R2(result.x, f, X_cv, Y_cv) 
                    if r2 > best_R2:
                        best_alpha = alpha
                        best_R2 = r2

                result = minimize(error, np.zeros(X.shape[0]+1), args=(X, Y, X_deriv, Y_deriv, best_alpha))
                P = result.x
                return {'weights' : P[1:],
                        'offset'  : P[0],
                        'params'  : P,
                        'error'   : result.fun,
                        'alpha'   : best_alpha,
                        'R2_train': R2(P, f, X, Y),
                        'R2_test' : R2(P, f, Xtest, Ytest)
                        }

        else:
            if not options['derivative_penalty']:
                best_neuron_idx = -1
                best_neuron_R2 = -np.inf
                best_neuron_params = np.zeros(2)
                best_neuron_error = np.inf

                for i in np.arange(X.shape[0]):
                    neuron = X[i,:]
                    result = minimize(error, np.zeros(2), args = (neuron, Y, 0))
                    r2 = R2(result.x, f, neuron, Y)
                    if r2 > best_neuron_R2:
                        best_neuron_idx = i
                        best_neuron_params = result.x
                        best_neuron_R2 = r2
                        best_neuron_error = result.fun
                
                weights = np.zeros(X.shape[0])
                weights[best_neuron_idx] = best_neuron_params[1]
                P = np.zeros(X.shape[0]+1)
                P[0] = best_neuron_params[0]
                P[1:] = weights

                return {'weights' : weights,
                        'offset'  : best_neuron_params[0],
                        'params'  : P,
                        'error'   : best_neuron_error,
                        'alpha'   : 0,
                        'R2_train': best_neuron_R2,
                        'R2_test' : R2(best_neuron_params, f, Xtest[best_neuron_idx,:], Ytest)
                        }
            
            else:
                X_deriv = gaussian_filter(X, sigma = (0, options['sigma']), order=1)
                Y_deriv = gaussian_filter(Y, sigma = options['sigma'], order = 1)

                best_neuron_idx = -1
                best_neuron_R2 = -np.inf
                best_neuron_params = np.zeros(2)
                best_neuron_error = np.inf

                for i in np.arange(X.shape[0]):
                    neuron = X[i,:]
                    neuron_deriv = X_deriv[i,:]
                    result = minimize(error, np.zeros(2), args = (neuron, Y, neuron_deriv, Y_deriv, 0))
                    r2 = R2(result.x, f, neuron, Y)
                    if r2 > best_neuron_R2:
                        best_neuron_idx = i
                        best_neuron_params = result.x
                        best_neuron_R2 = r2
                        best_neuron_error = result.fun
                
                weights = np.zeros(X.shape[0])
                weights[best_neuron_idx] = best_neuron_params[1]
                P = np.zeros(X.shape[0]+1)
                P[0] = best_neuron_params[0]
                P[1:] = weights

                return {'weights' : weights,
                        'offset'  : best_neuron_params[0],
                        'params'  : P,
                        'error'   : best_neuron_error,
                        'alpha'   : 0,
                        'R2_train': best_neuron_R2,
                        'R2_test' : R2(best_neuron_params, f, Xtest[best_neuron_idx,:], Ytest)
                        }
    
    else:
        clf = train_tree(X.T, Y)
        decision = clf.predict(X.T)

        X1 = X[:,decision > 0]
        X2 = X[:,decision < 0]
        Y1 = Y[decision > 0]
        Y2 = Y[decision < 0]

        new_options = deepcopy(options)
        new_options['decision_tree'] = False
        res1 = optimize_slm(X1, Y1, new_options)
        res2 = optimize_slm(X2, Y2, new_options)

        prediction_train = np.zeros(Y.size)
        prediction_train[decision > 0] = np.dot(X1.T, res1['weights']) + res1['offset']
        prediction_train[decision < 0] = np.dot(X2.T, res2['weights']) + res2['offset']

        decision_test = clf.predict(Xtest.T)
        prediction_test = np.zeros(Ytest.size)
        prediction_test[decision_test > 0] = np.dot(Xtest[:,decision_test>0].T, res1['weights']) + res1['offset']
        prediction_test[decision_test < 0] = np.dot(Xtest[:,decision_test<0].T, res2['weights']) + res2['offset']

        r2_train = 1-np.sum(np.power(Y-prediction_train, 2))/np.sum(np.power(Y-np.mean(Y), 2))
        r2_test = 1-np.sum(np.power(Ytest-prediction_test, 2))/np.sum(np.power(Ytest-np.mean(Ytest), 2))

        return {'clf'        : clf,
                'weights_pos': res1['weights'],
                'weights_neg': res2['weights'],
                'offset_pos' : res1['offset'],
                'offset_neg' : res2['offset'],
                'error'      : (Y1.size*res1['error']+Y2.size*res2['error'])/Y.size,
                'alpha_pos'  : res1['alpha'],
                'alpha_neg'  : res2['alpha'],
                'R2_train'   : r2_train,
                'R2_test'    : r2_test
                }

if __name__ == '__main__':
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(userTracker.codePath(), "regression_test.pdf"))

    for typ_cond in ['AML32_moving']:#, 'AML70_chip', 'AML70_moving', 'AML18_moving']:
        path = userTracker.dataPath()
        folder = os.path.join(path, '%s/' % typ_cond)
        dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

        # data parameters
        dataPars = {'medianWindow': 0,  # smooth eigenworms with gauss filter of that size, must be odd
                'gaussWindow': 50,  # gaussianfilter1D is uesed to calculate theta dot from theta in transformEigenworms
                'rotate': False,  # rotate Eigenworms using previously calculated rotation matrix
                'windowGCamp': 5,  # gauss window for red and green channel
                'interpolateNans': 6,  # interpolate gaps smaller than this of nan values in calcium data
                'volumeAcquisitionRate': 6.,  # rate at which volumes are acquired
                }
        dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars)
        keyList = np.sort(dataSets.keys())

        for key in keyList[:1]:
            time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
            neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
            velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']
            curvature = dataSets[key]['Behavior_crop_noncontig']['Eigenworm3']

            settings = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
            r2s = {}
            for s in settings:
                res = optimize_slm(neurons, velocity, options = {'derivative_penalty': s[0], 'decision_tree': s[1], 'best_neuron': s[2]})
                r2s[s] = (res['R2_train'], res['R2_test'])
                print(s, res['R2_train'], res['R2_test'])

            import pickle
            with open('all_regression_results.dat', 'wb') as handle:
                pickle.dump(r2s, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            fig, ax = plt.subplots(1, 1, figsize=(15, 12))

            r2keys = list(r2s.keys())
            labels = list(map(str,r2keys))
            r2s_train = [r2s[k][0] for k in r2keys]
            r2s_test  = [r2s[k][1] for k in r2keys]

            x = np.arange(len(r2s))  # the label locations
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, r2s_train, width, label=r'$R^2_{\mathrm{train}}$')
            rects2 = ax.bar(x + width/2, r2s_train, width, label=r'$R^2_{\mathrm{test}}$')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('$R^2$')
            ax.set_title(typ_cond+" "+key)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()


            def autolabel(rects):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')


            autolabel(rects1)
            autolabel(rects2)

            fig.tight_layout()
            pdf.savefig(fig)
    
    pdf.close()