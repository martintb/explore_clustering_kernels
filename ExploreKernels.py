import warnings
from itertools import product
import io
import numpy as np
import matplotlib.pyplot as plt
import gpflow
from sklearn.metrics import pairwise
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
import tqdm
import multiprocessing
from scipy.linalg import LinAlgError


import sys; sys.path.insert(0,'../')
import AFL
from DataDaemon import DataDaemon

import scipy.spatial

def mpl_figure_to_bytestring():
    with io.BytesIO() as f:
        plt.savefig(f)
        f.seek(0)
        bytestring = f.read()
    plt.close()
    plt.cla()
    plt.clf()
    return bytestring

def delaunay_adjacency(x):
    """
    Computes the Delaunay triangulation of the given points
    :param x: array of shape (num_nodes, 2)
    :return: the computed adjacency matrix
    """
    tri = scipy.spatial.Delaunay(x)
    edges_explicit = np.concatenate((tri.vertices[:, :2],
                                     tri.vertices[:, 1:],
                                     tri.vertices[:, ::2]), axis=0)
    adj = np.zeros((x.shape[0], x.shape[0]))
    adj[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.
    return np.clip(adj + adj.T, 0, 1) 


def process_params(kw):
    pm = kw['pm']
    
    ## Spectral Embedding 
    try:
        W1 = pairwise.pairwise_kernels(pm.measurements.values.copy(),
                                       metric=kw['affinity'],
                                       filter_params=True,
                                       gamma=kw['gamma'],
                                       degree=kw['degree'],
                                       coef0=kw['c0'])
    except ValueError as e:
        return
    
    if kw['co_affinity']=='distance':
        try:
            W2 = pairwise.pairwise_kernels(
                kw['distances'],
                metric='rbf',
                filter_params=True,
                gamma=kw['co_gamma']
            )
        except ValueError as e:
            return
    elif kw['co_affinity']=='adjacency':
        W2 = kw['adjacency']
    elif kw['co_affinity']=='None':
        W2 = 1.0
    else:
        return
        
    W = W1*W2
    
    n_components = len(pm.labels.unique())
    
    ## Clustering
    if kw['method']=='spectral_clustering':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = SpectralClustering(
                n_clusters=n_components, 
                affinity = 'precomputed', 
                assign_labels="discretize", 
                random_state=0, 
                n_init = 1000)
            
            try:
                clf.fit(W)
            except (ValueError,LinAlgError) as e:
                return
        labels = clf.labels_
    elif kw['method']=='gaussian_mixture_model':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GaussianMixture(n_components=n_components)
            try:
                clf.fit(W)
            except (ValueError,LinAlgError) as e:
                return
            labels = clf.predict_proba(W).argmax(1)
    else:
        return
        
    pm_l = pm.copy(labels=labels)
    fms = pm.fms(pm_l)
    
    
    # ## Plot Ground Truth
    # ax0 = pm_l.view.make_axes((1,1))
    # pm.plot(ax=ax0)
    # ax0.set_title(f'Ground Truth')
    # ground_truth_plot = mpl_figure_to_bytestring()
        
    ## Plot Predicted and convert to bytestring
    ax1 = pm_l.view.make_axes((1,1))
    pm_l.plot(ax=ax1)
    ax1.set_title(f'Predicted')
    result_plot = mpl_figure_to_bytestring()
    
        
    kw['queue'].put((
        [
            kw['npts'],
            kw['noise'],
            kw['method'],
            kw['affinity'],
            kw['co_affinity'],
            kw['gamma'],
            kw['degree'],
            kw['c0'],
            kw['co_gamma'],
            fms,
        ],
        result_plot
    ))
    

    
if __name__=='__main__':
    
    data_daemon = DataDaemon(db_name='db/results4.db',overwrite=False)
    
    npts_list = [325,1275]
    npts_list = [21,55,325,1275]
    pm_noise = {}
    plots = []
    for npts in npts_list:
        for noise in [0.0,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0]:
            pm = AFL.TernaryPhaseMap.load(f'noisy_data/pm-{noise}-{npts}.pkl')
            pm_noise[npts,noise] = pm
            
            ax0 = pm.view.make_axes((1,1))
            pm.plot(ax=ax0)
            ax0.set_title(f'Ground Truth')
            ground_truth_plot = mpl_figure_to_bytestring()
            data_daemon.add_ground_truth_plot(npts,ground_truth_plot)
            
    #start data_daemon process
    data_daemon.start(chunksize=100)
           
    
    affinity_params = {}
    affinity_params['chi2'] = ('gamma')
    affinity_params['linear'] = ()
    affinity_params['poly'] = ('gamma','degree','c0')
    affinity_params['rbf'] = ('gamma')
    affinity_params['laplacian'] = ('gamma')
    affinity_params['sigmoid'] = ('gamma','c0')
    affinity_params['cosine'] = ()

    affinities = [
        'chi2', 
        'poly', 
        'rbf', 
        'laplacian', 
        'sigmoid', 
        'cosine']
    
    co_affinities=['distance','adjacency','None']
    
    gamma_list = [1e-5,1e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1e0]
    degree_list = [0.5,1,1.5,2,3,4]
    c0_list = [-1000,-100,-10,-1,0,1,10,100,1000]
    method_list = ['spectral_clustering','gaussian_mixture_model']
    
    params = []
    total = 0
    for (npts,noise),pm in pm_noise.items():
        xy = pm.comp2cart()
        distances = pairwise.pairwise_distances(xy)
        adjacency = delaunay_adjacency(xy)
        for affinity,co_affinity,method in product(affinities,co_affinities,method_list):
            aff_parm = affinity_params[affinity]
            
            sub_parms = []
            if 'gamma' in aff_parm:
                sub_parms.append(gamma_list)
            else:
                sub_parms.append([-1])
                
            if 'degree' in aff_parm:
                sub_parms.append(degree_list)
            else:
                sub_parms.append([-1])
                
            if 'c0' in aff_parm:
                sub_parms.append(c0_list)
            else:
                sub_parms.append([-1])
            
            if co_affinity=='distance':
                sub_parms.append(gamma_list)
            else:
                sub_parms.append([-1])
                
                
            for gamma,degree,c0,co_gamma in product(*sub_parms):
                total+=1
                ### too slow :(
                # exists = data_daemon.exists((npts,noise,method,affinity,co_affinity,gamma,degree,c0,co_gamma))
                # if exists:
                #     continue
                    
                params.append(dict(
                    queue=data_daemon.queue,
                    npts = npts,
                    noise = noise,
                    method = method,
                    pm = pm,
                    affinity = affinity,
                    co_affinity = co_affinity,
                    distances = distances,
                    adjacency = adjacency,
                    gamma = gamma,
                    degree = degree,
                    c0 = c0,
                    co_gamma = co_gamma,
                ))
                
    
    #actually process the parameters
    p = multiprocessing.Pool(8)
    for _ in tqdm.tqdm(p.imap_unordered(process_params,params),total=len(params)):
        pass#map blocks
            
    data_daemon.stop()
        
    
    
    
    
    
    
    
    
    
    
    
    
    