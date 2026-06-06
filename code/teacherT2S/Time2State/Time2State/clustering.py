

   

from sklearn import mixture

import numpy as np

               

                                               

                                                                                

from sklearn import cluster

                                              

from Time2State.abstractions import *

class GMM(BasicClusteringClass):

    def __init__(self, n_states):

        self.n_states = n_states

    def fit(self, X):

        gmm = mixture.GaussianMixture(n_components=self.n_states, covariance_type="full").fit(X)

        return gmm.predict(X)

class GHMM(BasicClusteringClass):

    def __init__(self, n_component):

        self.n_component = n_component

    def fit(self, X):

        model = GaussianHMM(n_components=self.n_component, covariance_type='diag', n_iter=10000)

        model.fit(X)

        prediction = model.decode(X, algorithm='viterbi')[1]

        return prediction

class GMM_HMM(BasicClusteringClass):

    def __init__(self, n_states):

        self.n_states = n_states

        

    def fit(self, X):

        model = GMMHMM(n_components=self.n_states, covariance_type='diag', n_iter=10000)

        model.fit(X)

        prediction = model.decode(X, algorithm='viterbi')[1]

        return prediction

class DPGMM(BasicClusteringClass):

    def __init__(self, n_states, alpha=1e3):

        self.alpha = alpha

        if n_states is not None:

            self.n_states = n_states

        else:

            self.n_states = 20

    def fit(self, X):

        dpgmm = mixture.BayesianGaussianMixture(init_params='kmeans',

                                                n_components=self.n_states,

                                                covariance_type="full",

                                                weight_concentration_prior=self.alpha,        

                                                weight_concentration_prior_type='dirichlet_process',

                                                max_iter=1000).fit(X)

        return dpgmm.predict(X)

class KMeansClustering(BasicClusteringClass):

    def __init__(self, n_component):

        self.n_component = n_component

    def fit(self, X):

        clust = cluster.KMeans(n_clusters=self.n_component).fit(X)

        return clust.labels_

class SpectralClustering_(BasicClusteringClass):

    def __init__(self, n_component):

        self.n_component = n_component

    def fit(self, X):

        clust = cluster.SpectralClustering(n_clusters=self.n_component).fit(X)                                            

        return clust.labels_
