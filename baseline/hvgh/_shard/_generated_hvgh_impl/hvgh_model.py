
from __future__ import annotations

import os
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

try:
    from .hvgh_gp import GP
    from .hvgh_logsumexp import logsumexp
except Exception:
    from hvgh_gp import GP
    from hvgh_logsumexp import logsumexp

class Variational_Auto_Encoder():
  def __init__(self, input_dim, hidden_dims, latent_dim, kld_weight, epochs):
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.hidden_encoder_dim1 = hidden_dims[0]
    self.hidden_encoder_dim2 = hidden_dims[1]
    self.hidden_decoder_dim1 = hidden_dims[2]
    self.hidden_decoder_dim2 = hidden_dims[3]
    self.kld_weight = kld_weight
    self.opt = Adam(learning_rate=0.0001)
    self.epochs = epochs


    logvar_prior = tf.keras.Input(shape=(self.latent_dim, ), name='logvar_prior')
    mu_prior = tf.keras.Input(shape=(self.latent_dim, ), name='mu_prior')
    inputs = tf.keras.layers.Input(shape=(self.input_dim, ), name='encoder_input')
    hidden1= tf.keras.layers.Dense(self.hidden_encoder_dim1, activation='relu', name='enc1') (inputs)
    hidden2 = tf.keras.layers.Dense(self.hidden_encoder_dim2, activation='relu', name='enc2') (hidden1)
    z_mean = tf.keras.layers.Dense(self.latent_dim, activation='linear', name='z_mean')(hidden2)
    z_log_var= tf.keras.layers.Dense(self.latent_dim, activation='linear', name='z_log_var')(hidden2)
    z = tf.keras.layers.Lambda(self.sampling, name='z')([z_mean, z_log_var])

    enc_outputs = [z_mean, z_log_var, z]
    encoder = tf.keras.models.Model(inputs, enc_outputs, name='encoder')


    latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,), name='z_sampling')
    dec_hidden1 = tf.keras.layers.Dense(self.hidden_decoder_dim1, activation='relu', name='dec1') (latent_inputs)
    dec_hidden2 = tf.keras.layers.Dense(self.hidden_decoder_dim2, activation='relu', name='dec2') (dec_hidden1)
    outputs = tf.keras.layers.Dense(self.input_dim, activation='sigmoid') (dec_hidden2)

    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')


    inputs_ = [inputs, logvar_prior, mu_prior]
    outputs_ = [ decoder(encoder(inputs)[2]), encoder(inputs)[0], encoder(inputs)[1] , encoder(inputs)[2]]
    self.VAE = tf.keras.models.Model(inputs_, outputs_, name='VAE')


    MSE = tf.reduce_sum( tf.math.squared_difference(K.flatten(outputs_[0]), K.flatten(inputs_[0])))
    KLD = - 0.5 * tf.reduce_sum(1 + logvar_prior + z_log_var
            - (tf.pow(z_mean - mu_prior, 2)
            + tf.exp(z_log_var))/tf.exp(logvar_prior))
    loss = tf.reduce_mean(MSE + KLD * self.kld_weight )

    self.VAE.add_loss(loss)

  def sampling(self, args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon

  def compile(self):
    self.VAE.compile(optimizer=self.opt)


  def learn(self, data, logvar_prior, mu_prior, verbose=True):
    result = self.VAE.fit([data, logvar_prior, mu_prior] , epochs=1, verbose=verbose)
    return result

  def predict(self, data, logvar_prior, mu_prior, losses=False):
    reconst, mu, sigma, z = self.VAE.predict([data, logvar_prior, mu_prior])
    return reconst, mu, sigma, z

  def plot(self, data, reconst, mu, sigma, z, losses, savepath):
    if losses != False:
      plt.title('loss')
      plt.plot(np.arange(self.epochs), losses)
      plt.savefig(savepath+'_loss.png')
      plt.close()

    plt.title("z_alldim")
    plt.plot(np.arange(mu.shape[0]), mu)
    plt.savefig(savepath+'_z.png')
    plt.close()

    plt.title("z_hat_alldim")
    plt.plot(np.arange(mu.shape[0]), z)
    plt.savefig(savepath+'_z_hat.png')
    plt.close()

    plt.title('data_alldim')
    plt.plot(np.arange(mu.shape[0]), data)
    plt.savefig(savepath+'_oridata.png')
    plt.close()

    plt.title('reconst_alldim')
    plt.plot(np.arange(mu.shape[0]), reconst)
    plt.savefig(savepath+'_reconst.png')
    plt.close()

class GPMD:
    def __init__(self, dim):
        self.__dim = dim
        self.__gp = [ GP() for d in range(self.__dim) ]

    def learn(self,x, y ):
        y = np.array(y, dtype=float).reshape( (-1,self.__dim) )
        x = np.array(x,dtype=float)

        for d in range(self.__dim):
            if len(y)!=0:
                self.__gp[d].learn( x, y[:,d] )
            else:
                self.__gp[d].learn( x, [] )


    def calc_lik(self, x, y, last = False):
        lik = 0.0
        mus = []
        sigmas = []

        if self.__dim==1:
            y = np.asarray(y, dtype=float).reshape( (-1,self.__dim) )
        for d in range(self.__dim):
            lik += self.__gp[d].calc_lik( x , y[:,d] )
            if last != False:
              mu , sig = self.__gp[d].predict(x)
              mus.append(mu)
              sigmas.append(sig)

        if last != False:
          return lik, np.array(mus, dtype=float), np.array(sigmas, dtype=float)
        else:
          return lik

    def plot(self, x ):
        for d in range(self.__dim):
            plt.subplot( self.__dim, 1, d+1 )

            mus, sigmas = self.__gp[d].predict(x)
            y_min = mus - sigmas*2
            y_max = mus + sigmas*2

            plt.fill_between( x, y_min, y_max, facecolor="lavender" , alpha=0.9 , edgecolor="lavender"  )
            plt.plot(x, y_min, 'b--')
            plt.plot(x, mus, 'b-')
            plt.plot(x, y_max, 'b--')

class GPSegmentation():
    def __init__(self, dim, gamma, alpha, initial_class):
        self.dim = dim
        self.numclass = initial_class
        self.segmlen = 3
        self.gps = [ GPMD(dim) for i in range(self.numclass) ]
        self.segm_in_class= [ [] for i in range(self.numclass) ]
        self.segmclass = {}
        self.segments = []
        self.trans_prob = np.ones( (1,1) )
        self.trans_prob_bos = np.ones( 1 )
        self.trans_prob_eos = np.ones( 1 )
        self.all_numclass = []
        self.counter = 0
        self.is_initialized = False


        self.MAX_LEN = 20
        self.MIN_LEN = 3
        self.AVE_LEN = 12
        self.SKIP_LEN = 1

        self.alpha = alpha
        self.beta = np.ones(1)
        self.gamma = gamma

    def load_data(self, zs, classfile=None ):
        self.data = []
        self.segments = []
        self.is_initialized = False

        for y in zs:
            segm = []
            self.data.append( np.array(y, dtype=float) )

            i = 0
            while i<len(y):
                length = random.randint(self.MIN_LEN, self.MAX_LEN)

                if i+length+1>=len(y):
                    length = len(y)-i

                segm.append( y[i:i+length+1] )

                i+=length

            self.segments.append( segm )

            for i,s in enumerate(segm):
                c = random.randint(0,self.numclass-1)
                self.segmclass[id(s) ] = c

        self.calc_trans_prob()


    def load_model( self, basename ):
        for c in range(self.numclass):
            filename = basename + "class%03d.npy" % c
            self.segm_in_class[c] = np.load( filename, allow_pickle=True)
            self.update_gp( c )

        self.trans_prob = np.load( basename+"trans.npy", allow_pickle=True )
        self.trans_prob_bos = np.load( basename+"trans_bos.npy", allow_pickle=True )
        self.trans_prob_eos = np.load( basename+"trans_eos.npy", allow_pickle=True )


    def update_gp(self, c ):
        datay = []
        datax = []
        for s in self.segm_in_class[c]:
            datay += [ y for y in s ]
            datax += range(len(s))

        self.gps[c].learn( datax, datay )


    def calc_emission_logprob( self, c, segm ):
        gp = self.gps[c]
        slen = len(segm)

        if len(segm) > 2:
            log_plen = (slen*math.log(self.AVE_LEN) + (-self.AVE_LEN)*math.log(math.e)) - (sum(np.log(np.arange(1,slen+1))))
            p = gp.calc_lik( np.arange(len(segm), dtype=float) , segm )
            return p + log_plen
        else:
            return math.log(1.0e-100)

    def save_model(self, basename ):
        if not os.path.exists(basename):
            os.mkdir( basename )

        for n,segm in enumerate(self.segments):
            classes = []
            cut_points = []
            for s in segm:
                c = self.segmclass[id(s)]
                classes += [ c for i in range(len(s)) ]
                cut_points += [0] * len(s)
                cut_points[-1] = 1
            np.savetxt( basename+"segm%03d.txt" % n, np.vstack([classes,cut_points]).T, fmt=str("%d") )

        for c in range(len(self.gps)):
            for d in range(self.dim):
                plt.clf()
                for data in self.segm_in_class[c]:
                    if self.dim==1:
                        plt.plot( range(len(data)), data, "o-" )
                    else:
                        plt.plot( range(len(data[:,d])), data[:,d], "o-" )
                    plt.ylim( -1, 1 )

                plt.close()

        np.save( basename + "trans.npy" , self.trans_prob  )
        np.save( basename + "trans_bos.npy" , self.trans_prob_bos )
        np.save( basename + "trans_eos.npy" , self.trans_prob_eos )
        np.save( basename + "all_class.npy", self.segm_in_class[c])




        return self.numclass


    def forward_filtering(self, d ):
        T = len(d)
        log_a = np.log( np.zeros( (len(d), self.MAX_LEN, self.numclass) )  + 1.0e-100 )
        valid = np.zeros( (len(d), self.MAX_LEN, self.numclass) )
        z = np.ones( T )

        for t in range(T):
            for k in range(self.MIN_LEN,self.MAX_LEN,self.SKIP_LEN):
                if t-k<0:
                    break

                segm = d[t-k:t+1]
                for c in range(self.numclass):
                    out_prob = self.calc_emission_logprob( c, segm )
                    foward_prob = 0.0

                    tt = t-k-1
                    if tt>=0:
                        foward_prob = logsumexp( log_a[tt,:,:] + z[tt] + np.log(self.trans_prob[:,c]) ) + out_prob
                    else:
                        foward_prob = out_prob + math.log(self.trans_prob_bos[c])

                    if t==T-1:
                        foward_prob += math.log(self.trans_prob_eos[c])

                    log_a[t,k,c] = foward_prob
                    valid[t,k,c] = 1.0
                    if math.isnan(foward_prob):
                        print( "a[t=%d,k=%d,c=%d] became NAN!!" % (t,k,c) )
                        sys.exit(-1)

            if t-self.MIN_LEN>=0:
                z[t] = logsumexp( log_a[t,:,:] )
                log_a[t,:,:] -= z[t]

        return np.exp(log_a)*valid


    def sample_idx(self, prob ):
        accm_prob = [0,] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i


    def backward_sampling(self, a, d):
        T = a.shape[0]
        t = T-1

        segm = []
        segm_class = []

        c = -1
        while True:
            if t==T-1:
                transp = self.trans_prob_eos
            else:
                transp = self.trans_prob[:,c]

            idx = self.sample_idx( (a[t]*transp).reshape( self.MAX_LEN*self.numclass ))

            k = int(idx/self.numclass)
            c = idx % self.numclass

            if t-k-1<=0:
                s = d[0:t+1]
            else:
                s = d[t-k:t+1]

            segm.insert( 0, s )
            segm_class.insert( 0, c )

            t = t-k-1

            if t<=0:
                break

        return segm, segm_class


    def calc_trans_prob( self ):
        self.trans_prob = np.zeros( (self.numclass,self.numclass) )
        self.trans_prob_bos = np.zeros( self.numclass )
        self.trans_prob_eos = np.zeros( self.numclass )

        for n,segm in enumerate(self.segments):
            if id(segm[0]) in self.segmclass:
                c_begin = self.segmclass[ id(segm[0]) ]
                self.trans_prob_bos[c_begin]+=1

            if id(segm[-1]) in self.segmclass:
                c_end = self.segmclass[ id(segm[-1]) ]
                self.trans_prob_eos[c_end]+=1

            for i in range(1,len(segm)):
                try:
                    cc = self.segmclass[ id(segm[i-1]) ]
                    c = self.segmclass[ id(segm[i]) ]
                except KeyError:

                    continue
                self.trans_prob[cc,c] += 1

        self.trans_prob_bos += self.alpha * self.beta
        self.trans_prob_eos += self.alpha * self.beta

        for c in range(self.numclass):
            self.trans_prob[c,:] += self.alpha * self.beta

        self.trans_prob = self.trans_prob / self.trans_prob.sum(1).reshape(self.numclass,1)
        self.trans_prob_bos = self.trans_prob_bos / np.sum( self.trans_prob_bos )
        self.trans_prob_eos = self.trans_prob_eos / np.sum( self.trans_prob_eos )


    def sample_num_states(self):


        u = []
        for n,segm in enumerate(self.segments):
            c = self.segmclass[ id(segm[0]) ]
            p = self.trans_prob_bos[c]
            u.append( random.random() * p )

            c = self.segmclass[ id(segm[-1]) ]
            p = self.trans_prob_eos[c]
            u.append( random.random() * p )

            for i in range(1,len(segm)):
                cc = self.segmclass[ id(segm[i-1]) ]
                c = self.segmclass[ id(segm[i]) ]
                p = self.trans_prob[cc,c]
                u.append( random.random() * p )


        beta = list( self.beta )
        for c in range(self.numclass)[::-1]:
            if len(self.segm_in_class[c])==0:
                self.numclass -= 1
                self.gps.pop()
                self.segm_in_class.pop()
                beta[-2] += beta[-1]
                beta.pop()

            else:
                break

        u_min = np.min( u )

        N = 0
        for c in range(self.numclass):
            N += len(self.segm_in_class[c])

        while self.alpha*beta[-1]/N > u_min:
            stick_len = beta[-1]
            rnd = np.random.beta(1,self.gamma)
            beta[-1] = stick_len * rnd
            beta.append( stick_len * (1-rnd) )
            self.numclass += 1
            self.gps.append( GPMD(self.dim) )
            self.segm_in_class.append([])

        self.beta = np.array( beta , dtype=float)

        self.all_numclass.append(self.numclass)



    def remove_ndarray(self, lst, elem ):
        l = len(elem)
        for i,e in enumerate(lst):
            if len(e)!=l:
                continue
            if (e==elem).all():
                lst.pop(i)
                return
        raise ValueError( "ndarray is not found!!" )

    def learn(self):
        if self.is_initialized==False:

            for i in range(len(self.segments)):
                for s in self.segments[i]:
                    c = self.segmclass[id(s)]
                    self.segm_in_class[c].append( s )


            for c in range(self.numclass):
                self.update_gp( c )

            self.is_initialized = True

        self.update(True)

    def recog(self):
        self.update(False)

    def update(self, learning_phase=True ):

        for i in range(len(self.segments)):
            if learning_phase:
                print ("slice sampling")
                self.sample_num_states()

            d = self.data[i]
            segm = self.segments[i]

            for s in segm:
                c = self.segmclass[id(s)]
                self.segmclass.pop( id(s) )

                if learning_phase:

                    self.remove_ndarray( self.segm_in_class[c], s )

            if learning_phase:

                for c in range(self.numclass):
                    self.update_gp( c )


                self.calc_trans_prob()



            a = self.forward_filtering( d )


            segm, segm_class = self.backward_sampling( a, d )








            self.segments[i] = segm

            for s,c in zip( segm, segm_class ):
                self.segmclass[id(s)] = c


                if learning_phase:
                    self.segm_in_class[c].append(s)

            if learning_phase:

                for c in range(self.numclass):
                    self.update_gp( c )


                self.calc_trans_prob()
        return


    def calc_lik(self, last=False):
        liks = 0
        mus_all = []
        sigmas_all = []

        for segm in self.segments:

            if last != False:
              mus = [[] for i in range(self.dim)]
              sigmas = [[] for i in range(self.dim)]

            for n, s in enumerate(segm):
                c = self.segmclass[id(s)]
                liks += self.gps[c].calc_lik( np.arange(len(s),dtype=float) , np.array(s, dtype=float) )


                if last != False:
                  lik, mu, sig = self.gps[c].calc_lik( np.arange(len(s), dtype=float) , s , last)
                  if n == 0:
                    for dd in range(self.dim):
                        mus[dd] = mu[dd]
                        sigmas[dd] = sig[dd]
                  else:
                    for dd in range(self.dim):
                        mus[dd] = np.concatenate([mus[dd], mu[dd]])
                        sigmas[dd] = np.concatenate([sigmas[dd], sig[dd]])

                  liks += lik


            if last != False:
              mus_all.append((np.array(mus, dtype=float).T).astype(dtype=float))
              sigmas_all.append(np.log((np.array(sigmas, dtype=float).T).astype(dtype=float)))


        if last != False:
          return liks, mus_all, sigmas_all
        else:
          return liks

    def get_num_class(self):
        n = 0
        for c in range(self.numclass):
            if len(self.segm_in_class[c])!=0:
                n += 1
        return n

def learn(zs, savedir, dim, gamma, eta, initial_class):
    gpsegm = GPSegmentation(dim, gamma, eta, initial_class)

    gpsegm.load_data( zs )
    liks = []


    for it in range(5):
        print( "-----", it, "-----" )
        gpsegm.learn()
        numclass = gpsegm.save_model(savedir)
        print("lik =", gpsegm.calc_lik())
        liks.append(gpsegm.calc_lik())







    lik, mu, sigma = gpsegm.calc_lik(last=True)
    return numclass, np.array(mu, dtype=float), np.array(sigma, dtype=float)




class HVGH():
    def __init__(self, epoch=1, iteration=1, gamma=2, eta=2, initial_class=1):
        self.epoch = epoch
        self.iteration = iteration
        self.gamma = gamma
        self.eta = eta
        self.initial_class = initial_class

    def fit(self, dta, save_dir, win_size=20, input_dim=None):
        hidden_dim = [40,20,20,40]
        latent_dim= 3


        kld_weight = 0.9

        data_ = []
        batch_sizes = []
        logvar_priors = []
        mu_priors = []

        y = dta[::win_size]
        data_.append( y )
        batch_sizes.append( int(len(y)/4) )

        logvar_priors.append( np.array( np.zeros( (len(y),latent_dim) ), dtype=float ) )
        mu_priors.append( np.array( np.zeros( (len(y),latent_dim) ), dtype=float ) )

        if input_dim is None:
            input_dim=len(data_[0][0])
        else:
            input_dim=input_dim


        gamma = self.gamma
        eta = self.eta
        initial_class = self.initial_class


        vae = Variational_Auto_Encoder(input_dim, hidden_dim, latent_dim, kld_weight, self.epoch)
        vae.compile()

        path = ('./')

        for ite in range(self.iteration):
            print ("*--------------iteration:%03d--------------*"%ite)
            zs = []
            for n, data in enumerate(data_):
                losses = []

                for e in range(self.epoch):
                    idx = np.random.choice(range(0, len(data)), batch_sizes[n])
                    result = vae.learn(data[idx], logvar_priors[n][idx], mu_priors[n][idx], verbose=0)
                    losses.append(result.history['loss'])


                reconst, mu, sigma, z = vae.predict(data, logvar_priors[n], mu_priors[n], losses)
                savepath = (path+'HVGHlearn/'+save_dir+'/%03d/'%ite)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                savepath_ = (savepath + 'data_%03d'%n)
                vae.plot(data, reconst, mu, sigma, z, losses, savepath_)
                print ('VAE learned', 'iteration:', ite, 'data:', n)

                zs.append(np.array(mu))




            z_dim = len(zs[0][0])
            recog_initial_class, mu_priors, logvar_priors = learn(zs, savepath, z_dim, gamma, eta, initial_class )
