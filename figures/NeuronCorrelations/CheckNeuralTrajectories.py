
import os
import numpy as np
import matplotlib.pyplot as plt
from prediction import userTracker
import prediction.dataHandler as dh
from seaborn import clustermap

# For data set 110803 (moving only)- frames 1-1465, AVA 33 and 16
#Goal is to plot neural trajectories projected into first three PCs

def plot_a_trajectory(ax, pc_traj, theta=0, phi=0, color='#1f77b4'):
    ax.view_init(theta, phi)
    ax.plot(pc_traj[:, 0], pc_traj[:, 1], pc_traj[:, 2], color=color)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

def plot_trajectories(pc_traj, imm_start_index, end_index, title='Neural State Space Trajectories', color='#1f77b4'):
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    row=2
    col=4
    for nplot in np.arange(col) + 1:
        theta, phi = np.random.randint(360), np.random.randint(360)
        ax1 = plt.subplot(row, col, nplot, projection='3d', title='immobile (%d, %d)' % (theta, phi) )
        plot_a_trajectory(ax1, pc_traj[imm_start_index:end_index,:], theta, phi, color)

        ax2 = plt.subplot(row, col, nplot+col, projection='3d', title='moving (%d, %d)' % (theta, phi))
        plot_a_trajectory(ax2, pc_traj[:imm_start_index,:], theta, phi, color)

    import prediction.provenance as prov
    prov.stamp(plt.gca(), .55, .15, __file__)
    return


for typ_cond in ['AKS297.51_transition']: #, 'AKS297.51_moving']:
    path = userTracker.dataPath()
    folder = os.path.join(path, '%s/' % typ_cond)
    dataLog = os.path.join(path,'{0}/{0}_datasets.txt'.format(typ_cond))

    # Hard coding in folder for old dataset:
    #folder = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/'
    #dataLog = '/projects/LEIFER/PanNeuronal/decoding_analysis/old_worm_data/Special_transition/Special_transition_datasets.txt'

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
    theDataset = '193044'
    transition = im_start = 950
    im_end = 2885


    for key in filter(lambda x: theDataset in x, keyList):
        print("Running "+key)
        time = dataSets[key]['Neurons']['I_Time_crop_noncontig']
        time_contig = dataSets[key]['Neurons']['I_Time']
        neurons = dataSets[key]['Neurons']['I_smooth_interp_crop_noncontig']
        neurons_withNaN = dataSets[key]['Neurons']['I_smooth'] # use this to find the untracked neurons after transition
        neurons_ZScore = dataSets[key]['Neurons']['ActivityFull'] # Z scored neurons to use to look at calcium traces
        velocity = dataSets[key]['Behavior_crop_noncontig']['AngleVelocity']

        # Only consider neurons that have timepoints present for at least 75% of the time during immobilization
        frac_nan_during_imm = np.true_divide(np.sum(np.isnan(neurons_withNaN[:, im_start:]), axis=1),
                                             neurons_withNaN[:, transition:].shape[1])
        valid_imm = np.argwhere(frac_nan_during_imm < 0.25)[:, 0]


        dset = dataSets[key]
        Iz = neurons_ZScore
        # Cluster on Z-scored interpolated data to get indices
        # Cluster only on the immobile portion; and only consider neurons prsent for both moving and immobile
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(dataSets[key]['Neurons']['Activity'][valid_imm,transition:])
        d = dendrogram(Z, no_plot=True)
        idx_clust = np.array(d['leaves'])

        imm_start_time = time_contig[im_start]
        imm_start_index = np.abs(time - imm_start_time).argmin() #Index is for Noncontig
        end_time = time_contig[im_end]
        end_index = np.abs(time - end_time).argmin() # INdex is for noncontig





    #### Plot heatmap and behavior for whoel recording
    fig = plt.figure(figsize=(18, 18), constrained_layout=False)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(ncols=1, nrows=5, figure=fig, height_ratios=[2, .7, .7, .7, .7], width_ratios=[5])
    ax = fig.add_subplot(gs[0, :])

    prcntile = 99.7
    num_Neurons=neurons_withNaN[valid_imm, :].shape[0]
    vmin = np.nanpercentile(neurons_withNaN[valid_imm, :], 0.1)
    vmax = np.nanpercentile(neurons_withNaN[valid_imm, :].flatten(), prcntile)
    pos = ax.imshow(neurons_withNaN[valid_imm[idx_clust], :], aspect='auto',
                    interpolation='none', vmin=vmin, vmax=vmax,
                    extent=[time_contig[0], time_contig[-1], -.5, num_Neurons - .5], origin='lower')
    ax.set_ylim(-.5, num_Neurons + .5)
    ax.set_yticks(np.arange(0, num_Neurons, 25))
    ax.set_xticks(np.arange(0, time_contig[-1], 60))
    # ax.set_title('I_smooth_interp_crop_noncontig_wnans  (smooth,  interpolated, common noise rejected, w/ large NaNs, mean- and var-preserved, outlier removed, photobleach corrected)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron')
    ax.set_xlim(0, end_time)
    from matplotlib import ticker

    cb = fig.colorbar(pos, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()

    AVA1 = AVAR = 72#36
    AVA2 = AVAL = 22 #126#23
    AVAR_ci = np.argwhere(valid_imm[idx_clust] == AVAR)
    AVAL_ci = np.argwhere(valid_imm[idx_clust] == AVAL)

    yt = ax.get_yticks()
    yt = np.append(yt, [AVAR_ci, AVAL_ci])
    ytl = yt.tolist()
    ytl[-2:-1] = ["AVAR", "AVAL"]
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl)



    axbeh = fig.add_subplot(gs[1, :], sharex=ax)
    axbeh.plot(dset['Neurons']['TimeFull'], dset['BehaviorFull']['AngleVelocity'], linewidth=1.5, color='k')
    fig.colorbar(pos, ax=axbeh)
    axbeh.axhline(linewidth=0.5, color='k')
    axbeh.set_xticks(np.arange(0, time_contig[-1], 60))
    axbeh.set_xlim(ax.get_xlim())
    axbeh.set_ylabel('Velocity')


    curv = dset['BehaviorFull']['Eigenworm3']
    axbeh = fig.add_subplot(gs[2, :], sharex=ax)
    axbeh.plot(dset['Neurons']['TimeFull'], curv, linewidth=1.5, color='brown')
    axbeh.set_ylabel('Curvature')
    fig.colorbar(pos, ax=axbeh)
    axbeh.axhline(linewidth=.5, color='k')
    axbeh.set_xticks(np.arange(0, time_contig[-1], 60))
    axbeh.set_xlim(ax.get_xlim())

    axava1 = fig.add_subplot(gs[3, :], sharex=ax)
    axava1.plot(time_contig, dataSets[key]['Neurons']['gRaw'][AVAL,:])
    axava1.plot(time_contig, neurons_withNaN[AVAL,:])
    axava1.set_xticks(np.arange(0, time_contig[-1], 60))
    axava1.set_ylabel('AVAL')
    fig.colorbar(pos, ax=axava1)
    axava1.set_xlim(ax.get_xlim())

    axava = fig.add_subplot(gs[4, :], sharex=ax)
    axava.plot(time_contig, dataSets[key]['Neurons']['gRaw'][AVAR,:])
    axava.plot(time_contig, neurons_withNaN[AVAR,:])
    axava.set_ylabel('AVAR')
    axava.set_xticks(np.arange(0, time_contig[-1], 60))
    fig.colorbar(pos, ax=axava)
    axava.set_xlim(ax.get_xlim())




    #Repeat on the derivatives a la Kato et al
    def take_deriv(neurons):
        from prediction.Classifier import rectified_derivative
        _, _, nderiv = rectified_derivative(neurons)
        return nderiv
    Neuro_dFdt = take_deriv(neurons[valid_imm[idx_clust], :]).T
    Neuro = np.copy(neurons[valid_imm[idx_clust], ]).T  # I_smooth_interp_nonctoig


    def center_and_scale_around_immmobile_portion(recording, imm_start_index, end_index, with_std=False):
        # subtract and rescale the whole recording  so that the mean during hte immobile portion is zero
        # and, optionally, so that the variance during immobile portion is 1
        from sklearn.preprocessing import StandardScaler
        mean_scale = StandardScaler(copy=True, with_mean=True, with_std=with_std)
        mean_scale.fit(recording[imm_start_index:end_index, :]) #calcluate mean and or variance based on immobile
        return mean_scale.transform(recording) #rescale based on whole recording

    Neuro_mean_sub = center_and_scale_around_immmobile_portion(Neuro, imm_start_index, end_index, with_std=False)
    Neuro_z = center_and_scale_around_immmobile_portion(Neuro, imm_start_index, end_index, with_std=True)
    Neuro_dFdt_mean_sub = center_and_scale_around_immmobile_portion(Neuro_dFdt, imm_start_index, end_index, with_std=False)
    Neuro_dFdt_z = center_and_scale_around_immmobile_portion(Neuro_dFdt, imm_start_index, end_index, with_std=True)

    def print_AVAs_weights_in_pcs(AVAL_ci, AVAR_ci, pca, label=''):
        from sklearn.decomposition import PCA
        print(label)
        print("AVAL weights:", pca.components_[:, AVAL_ci])
        print("AVAR weights:", pca.components_[:, AVAR_ci])
        print("AVAL ranks:", np.where(np.argsort(np.abs(pca.components_)) == AVAL_ci))
        print("AVAR ranks:", np.where(np.argsort(np.abs(pca.components_)) == AVAR_ci))
        return

    def project_into_immobile_pcs(recording, imm_start_index, end_index, AVAL_ci=None, AVAR_ci=None, label=''):
        # Plot neural state space trajectories in first 3 PCs
        # also reduce dimensionality of the neural dynamics.
        from sklearn.decomposition import PCA
        nComp = 3  # pars['nCompPCA']
        pca = PCA(n_components=nComp, copy=True)
        pcs = pca.fit(recording[imm_start_index:end_index, :]).transform(recording)
        if AVAL_ci is not None and AVAR_ci is not None:
            print_AVAs_weights_in_pcs(AVAL_ci, AVAR_ci, pca, label=label)
        del pca
        return np.copy(pcs)


    pcs = project_into_immobile_pcs(Neuro_mean_sub, imm_start_index, end_index, AVAL_ci, AVAR_ci, label='activity')
    pcs_z = project_into_immobile_pcs(Neuro_z, imm_start_index, end_index, AVAL_ci, AVAR_ci, label='z-score')
    pcs_dFdt = project_into_immobile_pcs(Neuro_dFdt_mean_sub, imm_start_index, end_index, AVAL_ci, AVAR_ci, label='deriv')
    pcs_dFdt_z = project_into_immobile_pcs(Neuro_dFdt_z, imm_start_index, end_index, AVAL_ci, AVAR_ci, label='deriv Z-scored')

    plot_trajectories(pcs, imm_start_index, im_end, key + '\n F  PCA (minimally processed)')
    plot_trajectories(pcs_z,  imm_start_index, im_end, key + '\n F  PCA (z-scored)')
    plot_trajectories(pcs_dFdt,  imm_start_index, im_end, key + '\n F  dF/dt PCA (minimally processed)', color="#ff7f0e")
    plot_trajectories(pcs_dFdt_z,  imm_start_index, im_end, key + '\n F  dF/dt PCA (z-scored)', color="#ff7f0e")



    offset = 25
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(dset['Neurons']['TimeFull'], dset['BehaviorFull']['AngleVelocity'], 'k', label='velocity')
    plt.axhline(color="black")
    plt.ylabel('Velocity')
    plt.subplot(3, 1, 2)
    plt.plot(time, pcs[:, 0], label='PC0')
    plt.plot(time, offset + pcs[:, 1], label='PC1')
    plt.plot(time, 2 * offset + pcs[:, 2], label='PC2')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(time, neurons[AVA1, :], label='Neuron %d' % AVAL_ci)
    plt.plot(time, neurons[AVA2, :], label='Neuron %d' % AVAR_ci)
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(dset['Neurons']['TimeFull'],  dset['BehaviorFull']['AngleVelocity'], 'k', label='velocity')
    plt.axhline(color="black")
    plt.ylabel('Velocity')
    plt.subplot(3, 1, 2)
    plt.plot(time, pcs_z[:, 0], label='PC0 z')
    plt.plot(time, offset + pcs_z[:, 1], label='PC1 z')
    plt.plot(time, 2*offset + pcs_z[:, 2], label='PC2 z')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(time, pcs_dFdt_z[:, 0], label='PC0 z dF/dT')
    plt.plot(time, offset+pcs_dFdt_z[:, 1], label='PC1 z dF/dT')
    plt.plot(time, 2*offset + pcs_dFdt_z[:, 2], label='PC2 z dF/dT')
    plt.legend()


    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(dset['Neurons']['TimeFull'], dset['BehaviorFull']['AngleVelocity'], 'k', label='velocity')
    plt.axhline(color="black")
    plt.xlim(time[imm_start_index], time[end_index])
    plt.ylabel('Velocity')
    plt.subplot(3, 1, 2)
    plt.plot(time,  pcs_dFdt[:, 0], label='PC0 dF/dT')
    plt.plot(time, offset + pcs_dFdt[:, 1], label='PC1 dF/dT')
    plt.plot(time, 2 * offset + pcs_dFdt[:, 2], label='PC2 dF/dT')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(time, Neuro_dFdt[:, AVA1].T, label='Neuron %d' % AVA1)
    plt.plot(time, Neuro_dFdt[:, AVA2].T, label='Neuron %d' % AVA2)
    plt.legend()




    ### Next it will be important to show that the neurons before and after transitions
    # are likely the same
    before = np.arange(700, 800)
    after = np.arange(1100, 1200)
    av_b = np.nanmean(dataSets[key]['Neurons']['rRaw'][valid_imm, before[0]:before[-1]], axis=1)
    av_a = np.nanmean(dataSets[key]['Neurons']['rRaw'][valid_imm, after[0]:after[-1]], axis=1)
    av_bprime = np.nanmean(dataSets[key]['Neurons']['rRaw'][valid_imm, before[0] - 400 : before[-1] - 400], axis=1)
    av_aprime = np.nanmean(dataSets[key]['Neurons']['rRaw'][valid_imm, after[0] + 400 : after[-1] + 400], axis=1)
    plt.figure()
    for k in np.arange(av_b.shape[0]):
        plt.plot([0, 1], [av_b[k], av_a[k]], 'ko-')
        plt.plot([3, 4], [av_bprime[k], av_b[k]], 'ko-')
        plt.plot([5, 6], [av_a[k], av_aprime[k]], 'ko-')
    plt.text(0, 600, 'med |diff| = %.1f' % np.nanmedian(np.abs(av_b - av_a)))
    plt.text(3, 600, 'med |diff| = %.1f' % np.nanmedian(np.abs(av_bprime - av_b)))
    plt.text(5, 600, 'med |diff| = %.1f' % np.nanmedian(np.abs(av_a - av_aprime)))
    labels = ['(700 to 800)', '(1100 to 1200)', '(300 to 400)', '(700 to 800)', '(1100 to 1200)', '(1500 to 1600)']
    plt.xticks([0, 1, 3, 4, 5, 6], labels)
    plt.title('Change in Mean raw RFP Values across different time windows')
    plt.ylabel('F')
    plt.xlabel('Averaging Window (Volumes)')

    diff = av_a - av_b
    print("Neurons that have RFP values that change a lott before and after transition. (top two increase; top two decrease; python indexing")
    print(valid_imm[np.argsort(diff)[[0, 1, -1, -2]]])

    print("Plotting.")
    plt.show()


print("Done")
