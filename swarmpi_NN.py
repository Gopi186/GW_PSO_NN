#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
from pycbc.waveform import get_fd_waveform
from pycbc.waveform import get_td_waveform
from pycbc.filter import highpass, lowpass_fir
import pycbc.filter 
from pycbc.psd import aLIGOZeroDetHighPower,  AdvVirgo, KAGRA, from_txt
from pycbc.detector import Detector, get_available_detectors
from pycbc.noise.gaussian import frequency_noise_from_psd
from pycbc.noise.gaussian import noise_from_psd
from pycbc import frame
from pycbc import waveform
from pycbc.types import TimeSeries, FrequencySeries, Array, float32, float64, complex_same_precision_as, real_same_precision_as
## Unix Don't know how it works on other systems!
import time
from datetime import datetime
import sys
import os
import shutil
#from mpi4py import MPI # if needed in the futre
import pickle
from scipy.fft import fft
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv
###############################################################################
## Class swarm
## definition
###############################################################################
class inj_par :
    m1min=10.0    ## Ms
    m1max=25.0    ## Ms
    m2min=10.0    ## Ms
    m2max=25.0    ## Ms
    s1zmin=-0.8
    s1zmax=0.8
    s2zmin=-0.8
    s2zmax=0.8
    dmin=300.0      ## Mpc
    dmax=1000.0     ## Mpc
    seed=None
    def __init__(self, waveform,flow=20, enable_spin=0, 
                    inj_seed=None, ifo='I1', search_dim=2,
                    sps=4096, tmax=10.0, noise_seed=None) :
        if (inj_seed != None):
            self.seed=inj_seed
            np.random.seed(self.seed)
        self.noise_seed=noise_seed
        self.mass1=np.random.uniform(self.m1min, self.m1max)
        self.mass2=np.random.uniform(self.m2min, self.mass1)
        self.coa_phase=np.random.uniform(0, 2*np.pi)
        self.ra=np.random.uniform(0,np.pi)
        self.dec=np.random.uniform(0, 2*np.pi)
        self.pol=np.random.uniform(0, 2*np.pi)
        ## uniform over distance
        ## self.distance=np.random.uniform(self.dmin, self.dmax)
        ## But uniform over volume is what we need
        x=np.random.uniform(self.dmin, self.dmax)
        y=np.random.uniform(self.dmin, self.dmax)
        z=np.random.uniform(self.dmin, self.dmax)
        self.distance=np.sqrt(x*x+y*y+z*z) ## would be slightly off with 
                                ## upper limit should be fine
        self.f_lower=flow
        self.search_dim=search_dim
        self.waveform=waveform
        self.sps=sps
        self.tmax=tmax
        self.ifo=ifo
        if enable_spin !=0 :
            self.enable_spin=1
            self.spin1z=np.random.uniform(self.s1zmin, self.s1zmax)
            self.spin2z=np.random.uniform(self.s2zmin, self.s2zmax)
        else :
            self.spin1z=0.0
            self.spin2z=0.0
        self.signal_stat=3        
        

#------------------Initialisation for swarm class --------------------------------
class swarm :
    '''
        class to simulate partcile swarm optimisation
        Variables 


        f_low =20.0        -> lower frequency limit, used in matched filtering
        min = [10.0, 10.0] -> lower limit on search parameters, it should be an
                               array of dimension of parameter space and sets 
                                the lower limit along each dimension
        max = [10.0, 10.0] -> upper limit on search parameters, it should be an
                                array of dimension of parameter space and sets 
                                the upper limit along each dimension
        vmin =[-1.0, -1.0] -> lower limit on velovity parameters, it should be an
                                array of dimension of parameter space and sets 
                                the lower limit along each dimension
        vmax =[1.0, 1.0] -> upper limit on velocity parameters, it should be an
                                array of dimension of parameter space and sets 
                                the upper limit along each dimension
        dim = 2           ->   dimension of search parameter space
        sps = 4*1024      ->   sampling rate for the signal 
        T_max = 5.0       ->   length of data segment in sec
        delta_f           ->   frequency resolution
        Np                ->   Number of particles in the swarm
        Nsteps            ->   Maximum number of steps taken by the swarm
        curr_step         ->   Number of steps already taken by the swarm
        tlen              ->   Number of samples in time domain
        flen              ->   Number of samples in the fequency domain
        alpha=0.4         ->   Alpha parameter of PSO, couple to inertia
        beta=0.3          ->   Beta parameter of PSO, couple to p_best
        gamma=0.4         ->   Gamma parameter of PSO, couple to g_best
        p_best            ->   Np x dim array
        g_best            ->   dim array
        pbest_corr        ->   Np array
        gbest_corr        ->   real
        x                 ->   position   Np x dim array
        v                 ->   velocity   Np x dim array
        distance          ->   distance im tamplate m Mpc 
        waveform_approximant='IMRPhenomC'  -> trivial
        snr              ->   function value at x,  Np array 
        radius            ->   mean distance from g_best  array of size steps
        DEBUG             ->   can be False or True
    '''
##########################################################################
    def __init__(self,sps, tmax, sim_insp_par, ifo='L1',flow=20, particles=200, steps=30, dim=2,swarms=5):
        '''
        Initialise the class
        particles -> number of particle in the swarm
        steps     -> maximum number of steps taken by the swarm
        '''
        self.f_low = flow                    # For template
        self.f_upper=1000.0
        self.sps= sps                     # 2 K sps
        self.dim = dim                       #search dimensions
        self.T_max = tmax                     #time duration of wave
        self.delta_f = 1.0/self.T_max        #frequency resolution
        self.Np=particles                        
        self.Nsteps=steps
        self.Ns=swarms
        self.curr_step=0
        self.tlen=int(self.sps*self.T_max)
        self.flen=int(self.tlen/2)+1
        self.alpha=0.4
        self.beta=0.3
        self.gamma=0.35
        self.distance=1.0                  ## MPC 
        self.trigger_time=0.0
        self.waveform_approximant='IMRPhenomPv2'
        self.ifo=ifo                    ##['H1', 'L1', 'V1', 'K1', 'LI']
                                           ## 'H1' -> LIGO Hanford
                                           ## 'L1' -> LIGO Livingstone
                                           ## 'V1' -> VIRGO
                                           ## 'K1' -> Kagra
                                           ## 'I1' -> LIGO India
        #self.det = Detector(self.ifo)
        
        self.norm=1.0
        self.DEBUG = False
        if self.dim ==2 :
            self.parameters_names=["Mass1", "Mass2"]
            self.min = [10.0, 10.0]        #minimum value of mass being scanned
            self.max = [90.0, 90]          #maximum value of mass being scanned
            self.vmin =[-1.0, -1.0]        ## Velocity limits
            self.vmax=[1.0, 1.0]           ## Velocity limits
        if self.dim == 3 :
            self.parameters_names=["Mass1", "Mass2", "S1z"]
            self.min = [10.0, 10.0, -0.9]#minimum value of mass being scanned
            self.max = [90.0, 90, 0.9]    #maximum value of mass being scanned
            self.vmin =[-1.0, -1.0, -0.1]       ## Velocity limits
            self.vmax=[1.0, 1.0, 0.1]           ## Velocity limits
        if self.dim == 4 :
            self.parameters_names=["Mass1", "Mass2", "S1z", "S2z"]
            self.min = [10.0, 10.0, -0.9, -0.9]#minimum value of mass being scanned
            self.max = [90.0, 90, 0.9, 0.9]#maximum value of mass being scanned
            self.vmin =[-1.0, -1.0, -0.1, -0.1]     ## Velocity limits
            self.vmax=[1.0, 1.0, 0.1, 0.1]        ## Velocity limits
        self.x=np.zeros((self.Np,self.dim))
        self.x[:,0] = np.random.uniform(self.min[0], self.max[0], self.Np)#initial position of each particle in terms of mass 1 and mass 2
        self.x[:,1] = np.random.uniform(self.min[1], self.x[:,0], self.Np)#initial position of each particle in terms of mass 1
        for kk in range(2, self.dim):
            self.x[:,kk] = np.random.uniform(self.min[kk], self.max[kk], self.Np)
        #self.p_best = np.zeros((self.Np,self.dim)) #array that stores best position for each particle
        self.v = np.random.uniform(self.vmin, self.vmax, (self.Np,self.dim))#initial velocity of each particle
        self.pbest = np.copy(self.x)#initialise personal best   to initial position for each particle
        self.pbest_snr= np.zeros(self.Np)#to store old correlation
        self.pbest_toa=np.zeros(self.Np)
        self.gbest=np.zeros(self.dim)
        self.gbest_snr = 0.0#variable to store best match
        self.gbest_toa = 0.0
        self.snr = np.zeros(self.Np)
        self.toa = np.zeros(self.Np)
        self.radius=np.zeros(self.Nsteps)
        self.gbest_snr_step=np.zeros(self.Nsteps)
        self.gbest_toa_step=np.zeros(self.Nsteps)
        ############################################################################# Neural Network variables
        self.Mc=np.zeros((self.Np, self.Nsteps))
        self.corr_data=np.zeros((self.Np, self.Nsteps))
        self.pmass1=np.zeros((self.Np, self.Nsteps))
        self.pmass2=np.zeros((self.Np, self.Nsteps))
        #############################################################################
        self.sim_insp_par=sim_insp_par
        self.evolve=self.evolve_standard_pso
        self.evolve_option=0
        if self.ifo == 'L1' or self.ifo == 'H1' or self.ifo == 'I1' :
            self.psd = aLIGOZeroDetHighPower(self.flen, self.delta_f, self.f_low)
        elif self.ifo == 'V1' :
            self.psd = AdvVirgo(self.flen, self.delta_f, self.flow)
        elif self.ifo == 'K1' :
            self.psd = KAGRA(self.flen, self.delta_f, self.flow)
        elif self.ifo == 'L2' or self.ifo == 'H2' :    ## L2 and H2 stands for 02 noise PSD     
            self.psd = from_txt(sim_insp_par.O2_PSD_file,self.flen, self.delta_f, self.f_low,False)
        else :
            raise Exception('Unknown IFO :'+self.ifo)
        
##end of __init__
##########################################################################
    def __del__(self):
        '''
            cleanup if this might not be needed, but then why not
        '''
        del self.x, self.v, self.psd, self.radius, self.snr
        del self.gbest_snr, self.gbest, self.pbest_snr, self.pbest
        del self.min, self.max, self.vmin, self.vmax
        del self.f_low, self.sps, self.dim,self.T_max,
        del self.delta_f,self.Np,self.Nsteps,self.curr_step,self.tlen
        del self.flen, self.alpha, self.beta,self.gamma,self.distance
        del self.trigger_time, self.waveform_approximant, self.ifo
        del self.gbest_snr_step
##########################################################################
    def compute_fitness(self, segment,row):
        '''
            This is the function which is optmised!
        '''
        for jj in range (self.Np):#Correlation foreach point
            temp0, temp1=self.generate_template_strain_freqdomain(jj,row)
            csnr0=pycbc.filter.matched_filter(temp0, segment, psd=self.psd,  low_frequency_cutoff=self.f_low,high_frequency_cutoff=self.f_upper)
            csnr1=pycbc.filter.matched_filter(temp1,segment , psd=self.psd,  low_frequency_cutoff=self.f_low,high_frequency_cutoff=self.f_upper)
            csnr0=csnr0*np.conj(csnr0)
            csnr1=csnr1*np.conj(csnr1)
            idx0=np.argmax(csnr0)
            idx1=np.argmax(csnr1)
            
            snr0=csnr0[idx0]
            snr1=csnr1[idx1]
            mp=0.5*np.sqrt(snr0+snr1)
            
            self.corr_data[jj,self.curr_step]=np.real(mp) ################################### storing SNR of each particle at a particular step, for NN
            #print('Match',self.corr_data)
            #mp, i = pycbc.filter.match(segment, temp, psd=self.psd, 
            #low_frequency_cutoff=self.f_low,high_frequency_cutoff=self.f_upper,
            #v1_norm=self.norm, v2_norm=None)
            #mpx= filter.matched_filter(segment, temp_series, psd=self.psd, low_frequency_cutoff=self.f_low)
            #if mp > 1e4 :
            #    print jj
            #    print self.x[jj,:]
            #    raise Exception('SNR blowup'+str(mp))
            
            self.snr[jj] = np.real(mp) # v1_norm=None, v2_norm=None
            self.toa[jj] =idx0
            #self.corr[jj] = mpx.abs_max_loc()[0]
            #print 'here=',mpx.abs_max_loc()
        
        #results=self.neural_net()
        #print(results, 'Results')
## end of compute_fitness
##########################################################################
    def compute_pbest(self):
        '''
            This function find the best postion for a given particle
            called p_best
        '''
        for jj in range (self.Np): ## Finds pbest
            if (self.snr[jj]>self.pbest_snr[jj]):
                self.pbest[jj,:] = self.x[jj,:]
                self.pbest_snr[jj] = self.snr[jj]
                self.pbest_toa[jj]=self.toa[jj]
#updating best match values obtained for each particle
## end of compute_pbest
##########################################################################
    def compute_gbest(self):
        '''
            This function find the global best position found by all particles
        '''
        idbest= np.argmax(self.pbest_snr)
        if self.pbest_snr[idbest] > self.gbest_snr :
            self.gbest[:]=self.pbest[idbest,:]
            self.gbest_snr=self.pbest_snr[idbest]
            self.gbest_toa=self.pbest_toa[idbest]
## end of compute_gbest
##########################################################################
    def evolve_velocity(self):
        '''
            This function evolve the velocity as per standard PSO rule
        '''
        r0 = np.random.rand()
        r1 = np.random.rand()
        r2 = np.random.rand()
        for jj in range (self.Np):
            self.v[jj,:] = self.alpha*r0*self.v[jj,:] +\
            self.beta*r1*(self.pbest[jj,:]-self.x[jj,:]) +\
            self.gamma*r2*(self.gbest-self.x[jj,:])
## end of evolve_velocity
##########################################################################
    def evolve_position(self):
        '''
            This evolves the position of the particle for a given velocity
            This function also updates the current step a variable keeps 
            track of number of steps taken
        '''
        for jj in range(self.Np):
        
            #print('Chirpmass', self.Mc)
            self.x[jj,:]=self.x[jj,:]+self.v[jj,:]
            ########################################################## storing position(mass pair) and calculating chirp mass of each particle for each step for NN
            self.pmass1[jj,self.curr_step]=np.copy(self.x[jj,0])
            self.pmass2[jj,self.curr_step]=np.copy(self.x[jj,1])
            self.Mc[jj,self.curr_step]=((self.pmass1[jj,self.curr_step]*self.pmass2[jj,self.curr_step])**(3/5))/((self.pmass1[jj,self.curr_step]+self.pmass2[jj,self.curr_step])**(1/5)) ####Chirp mass
            ##########################################################    
## Any boundary condition can be explicitly applied
        self.curr_step+=1
## end of evolve_position
##########################################################################
    def store_gbest(self) :
        '''
            This function stores gbest for that perticluar step
        '''
        self.gbest_snr_step[self.curr_step]=self.gbest_snr
        self.gbest_toa_step[self.curr_step]=self.gbest_toa
##########################################################################
    def compute_radius(self):
        '''
            This function computes the mean size, i.e. distance from the gbest of 
            swarm
        '''
        if self.DEBUG :
            print("This is function compute_radius", self.curr_step)
        rad=np.zeros(self.Np)
        for jj in range(self.Np):
            rad[jj]=np.sqrt(np.sum(np.square(self.x[jj,:]-self.gbest)))
        self.radius[self.curr_step]=np.mean(rad)
## end of compute_radius
##########################################################################
    def apply_reflecting_boundary(self):
        '''
            This function put the reflecting boundary at the wall, this might
            inprove the convergence if the peak is close to boundary! 
        '''
        
## Hope this reflextion about minimum mass for m1 is done
        for ii in range(self.dim) :
            idx=np.where(self.x[:,ii] < self.min[ii])
            self.x[idx,ii]=2*self.min[ii]-self.x[idx,ii]
            self.v[idx,ii]=-self.v[idx,ii]
            idx=np.where(self.x[:,ii] > self.max[ii])
            self.x[idx,ii]=2*self.max[ii]-self.x[idx,ii]
            self.v[idx,ii]=-self.v[idx,ii]
            idx=np.where(self.x[:,0] < self.x[:,1])
            self.x[idx,1]=2*self.x[idx,0]-self.x[idx,1]
            self.v[idx,0:2]=-self.v[idx,0:2]
            self.apply_velocity_condition()
            '''
            
            for ii in range (self.Np):#keeping direction same
                if (np.linalg.norm(self.v[ii,0:2])>15.0):
                    self.v[ii,0:2] = np.random.uniform(0.0,1.0)*self.v[ii,0:2]/np.linalg.norm(self.v[ii,0:2])
            if self.dim == 4:
                for ii in range (self.Np):
                    if (np.linalg.norm(self.v[ii,2:4])>0.25):
                        self.v[ii,2:4] = np.random.uniform(0.0,0.1)*self.v[ii,2:4]/np.linalg.norm(self.v[ii,2:4])
            
            '''
            
            '''
            if ii < 2:#velocity is kept between -15 and +15 in mass-mass plane
                idx = np.where(self.v[:,ii]<-15.0)
                self.v[idx,ii] = np.random.uniform(self.vmin[ii],0.0)#reset to number between -1 and 0
                idx = np.where(self.v[:,ii]>15.0)
                self.v[idx,ii] = np.random.uniform(0.0,self.vmax[ii])#reset to number between 0 and +1
                        
            if ii >= 2:#velocity is kept between -0.25 and +0.25 in spin-spin plane
                idx = np.where(self.v[:,ii]<-0.25)
                self.v[idx,ii] = np.random.uniform(self.vmin[ii],0.0)
                idx = np.where(self.v[:,ii]>0.25)
                self.v[idx,ii] = np.random.uniform(0.0,self.vmax[ii])
            '''
        
##########################################################################
##########################################################################
    def apply_velocity_condition(self):
        '''
            This function put the reflecting boundary at the wall, this might
            inprove the convergence if the peak is close to boundary! 
        '''
        ###############
        x1=self.x[:,0]
        y1=self.x[:,1]
        dx=np.abs(x1-y1)
        dv=self.v[:,0]+self.v[:,1]
        idx=np.where(dv > dx)
        scale=0.9
        #print  idx
        self.v[idx,0]=self.v[idx,0]/dv[idx]*dx[idx]*scale
        self.v[idx,1]=self.v[idx,1]/dv[idx]*dx[idx]*scale
        dx=np.abs(self.x[:,0]-self.max[0])
        dv=self.v[:,0]
        idx=np.where(dv > dx)
        self.v[idx,0]=self.v[idx,0]/dv[idx]*dx[idx]*scale
        dy=np.abs(self.x[:,1]-self.max[1])
        dv=self.v[:,1]
        idx=np.where(dv > dx)
        self.v[idx,1]=self.v[idx,1]/dv[idx]*dx[idx]*scale
        
        dx=np.abs(self.min[0]-self.x[:,0])
        dv=self.v[:,0]
        idx=np.where(dv > dx)
        self.v[idx,0]=self.v[idx,0]/dv[idx]*dx[idx]*scale
        dy=np.abs(self.min[1]-self.x[:,1])
        dv=self.v[:,1]
        idx=np.where(dv > dx)
        self.v[idx,1]=self.v[idx,1]/dv[idx]*dx[idx]*scale
        
##########################################################################
    def evolve_standard_pso(self, segment,row) :
        '''
            This function takes all it needed to take one step as per standard
            PSO
        '''
        self.compute_radius()
        self.compute_fitness(segment,row)
        self.compute_pbest()
        self.compute_gbest()
        self.store_gbest()
        self.evolve_velocity()
        self.evolve_position()
        self.apply_reflecting_boundary()
        return self.corr_data, self.pmass1, self.pmass2
        #self.neural_net()
## end of  evolve_standard_pso
##########################################################################
    def neural_net(self):
        inputs = keras.Input(shape=(1), name="correlation")
        x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
        #self.corr_data=corr_data
        model = keras.Model(inputs=inputs, outputs=outputs)
        x_train = self.corr_data[:,self.curr_step]
        y_train = np.arange(0,300,1)
        x_test= np.random.uniform(self.min[0], self.max[0], self.Np)
        y_test = np.arange(0,300,1)

        model.compile(
            optimizer=keras.optimizers.RMSprop(),  # Optimizer
            # Loss function to minimize
            loss=keras.losses.MeanSquaredError(),
            # List of metrics to monitor
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        print("Fit model on training data")
        history = model.fit(
            x_train,
            y_train,
           batch_size=64,
            epochs=2)
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
# Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = model.evaluate(x_test, y_test, batch_size=128)
        #print("test loss, test acc:", results)
        return results


##########################################################################
    def evolve_along_mass_eigen_direction(self, segment) :
        '''
            This function takes a step along one of the eigen value of
            the correlation matrix, 
            drxn -> x takes step along 'x' eigen-direction 
            ## God's comment! This needs some thought on extending
            ## higher dimension
            ## direction will be removed in the future version
            drxn = 0 means along both eigen vectors
            drxn = 1 along eignen vector with larger eigen value
            drxn = 2 along eignen vector with smaller eigen value
        '''
        drxn=self.evolve_option
        corr_mat=np.zeros((2,2))
        self.compute_radius()
        self.compute_fitness(segment)
        self.compute_pbest()
        self.compute_gbest()
        x0=np.copy(self.x)
        v0=np.copy(self.v)
        f0=np.copy(self.snr)
        self.evolve_velocity()
        self.evolve_position()
        self.apply_reflecting_boundary()
        self.compute_fitness(segment)
        self.compute_pbest()
        self.compute_gbest()
        ev=np.zeros(2)
        for ii in range(self.Np) :
            corr_mat[0,0]=f0[ii]
            corr_mat[1,1]=self.snr[ii]
            m1=x0[ii,0]; m2=self.x[ii,1]
            hptilde, hctilde = get_fd_waveform(approximant=self.waveform_approximant,
                                 mass1=m1,
                                 mass2=m2,
                                 distance=self.distance,
                                 f_lower=self.f_low,
                                 delta_f=self.delta_f)
            hptilde.resize(self.flen)
            #psd = aLIGOZeroDetHighPower(self.flen, self.delta_f, self.f_low)
            mp, i = filter.match(segment, hptilde, psd=self.psd,
                                    low_frequency_cutoff=self.f_low,
                                    v1_norm=self.norm, v2_norm=None)
            corr_mat[0,1]=mp
            if mp > self.gbest_snr :
                self.gbest_snr=mp
                self.gbest[:]=self.x[ii,:]
                self.gbest[0]=m1
                self.gbest[1]=m2
            
            m1=self.x[ii,0]; m2=x0[ii,1]
            hptilde, hctilde = get_fd_waveform(approximant=self.waveform_approximant,
                                 mass1=m1,
                                 mass2=m2,
                                 distance=self.distance,
                                 f_lower=self.f_low,
                                 delta_f=self.delta_f)
            hptilde.resize(self.flen)
                    # Generate the aLIGO ZDHP PSD
            mp, i = filter.match(segment, hptilde, psd=self.psd, 
                                    low_frequency_cutoff=self.f_low,
                                    v1_norm=self.norm, v2_norm=None)
            corr_mat[1,0]=mp
            if mp > self.gbest_snr :
                self.gbest_snr=mp
                self.gbest[:]=self.x[ii,:]
                self.gbest[0]=m1
                self.gbest[1]=m2
            #print corr_mat
            wa, va = np.linalg.eig(corr_mat)
            drxn1 = np.argmax(wa)                   #larger eigenvalue
            drxn2 = (drxn1+1)%2                   # Smaller eigenvalue :)
            
            ev0=va[:,drxn1]                    #larger eigenvector
            ev1=va[:,drxn2]                     #smaller eigenvector
            disp = self.x[ii,:]-x0[ii,:]        #difference in position
            vel = self.v[ii,:]-v0[ii,:]         #difference in velocity
            proj_disp0 = np.dot(disp, ev0)*ev0 #projection of displacement on larger e.v
            proj_disp1 = np.dot(disp, ev1)*ev1 #projection on smaller e.v
            proj_vel0 = np.dot(vel, ev0)*ev0   #projection of velocity difference on larger e.v
            proj_vel1 = np.dot(vel, ev1)*ev1   #projection on smaller e.v
            dm1=self.x[ii,0]-x0[ii,0]
            dm2=self.x[ii,1]-x0[ii,1]
            dv1=self.v[ii,0]-v0[ii,0]
            dv2=self.v[ii,1]-v0[ii,1]
            if drxn ==0 :
                self.x[ii,:] = self.x[ii,:] + proj_disp0 + proj_disp1
                #self.v[ii,:] = self.v[ii,:] + proj_vel0 + proj_vel1
                        ## This combination seems to converge! 
            if drxn ==1 :
                self.x[ii,:] = self.x[ii,:] + proj_disp0
                #self.v[ii,:] = self.v[ii,:] + proj_vel0
                ## This combination seems to converge too!
            if drxn ==2 :
                self.x[ii,:] = self.x[ii,:] + proj_disp1
                #self.v[ii,:] = self.v[ii,:] + proj_vel1

                #self.v[ii,:] = self.v[ii,:] - proj_vel0
                ## This combination seems to converge too!
            if drxn ==2 :
                self.x[ii,:] = self.x[ii,:] + proj_disp1
                #self.v[ii,:] = self.v[ii,:] - proj_vel1
                ## This combination seems to converge too!
            if drxn == 3:
                self.x[ii,0]=self.x[ii,0]-(dm1*va[0,0]+dm2*va[0,1])
                self.x[ii,1]=self.x[ii,1]-(dm1*va[1,0]+dm2*va[1,1])
            self.apply_reflecting_boundary()
        #print wa
        #print wa[drxn]*va[:,drxn]


        #self.v[ii,:]=self.v[ii,:]+wa[drxn]*va[:,drxn]
        ## end of evolve_along_eigen_direction
## end of evolve_along_eigen_direction
##########################################################################
##########################################################################
    def evolve_along_spin_eigen_direction(self, segment) :
        '''
            needs clean up, might not work correctly
            This function takes a step along one of the eigen value of
            the correlation matrix, 
            drxn -> x takes step along 'x' eigen-direction 
            ## God's comment! This needs some thought on extending
            ## higher dimension
            ## direction will be removed in the future version
        '''
        drxn=self.evolve_option
        corr_mat=np.zeros((2,2))
        self.compute_radius()
        self.compute_fitness(segment)
        self.compute_pbest()
        self.compute_gbest()
        x0=np.copy(self.x)
        v0=np.copy(self.v)
        f0=np.copy(self.snr)
        self.evolve_velocity()
        self.evolve_position()
        self.apply_reflecting_boundary()
        self.compute_fitness(segment)
        ev=np.zeros(2)
        for ii in range(self.Np) :
#print x0[ii,:], self.x[ii,:], x0[ii,:]-self.x[ii,:]
            corr_mat[0,0]=f0[ii]
            corr_mat[1,1]=self.snr[ii]
            s1z=x0[ii,2]; s2z=self.x[ii,3]
            ds=self.x[ii,2:4]-x0[ii,2:4]
            dv=self.v[ii,2:4]-v0[ii,2:4]

            hptilde, hctilde = get_fd_waveform(approximant=self.waveform_approximant,
                         mass1=self.x[ii,0],
                         mass2=self.x[ii,1],
                         spin1z=s1z,
                         spin2z=s2z,
                         distance=self.distance,
                         f_lower=self.f_low,
                         delta_f=self.delta_f)
            hptilde.resize(self.flen)
            psd = aLIGOZeroDetHighPower(self.flen, self.delta_f, self.f_low)
            mp, i = filter.match(segment, hptilde, psd=psd, low_frequency_cutoff=self.f_low)
            corr_mat[0,1]=mp
            s1z=x0[ii,3]; s2z=self.x[ii,2]
            hptilde, hctilde = get_fd_waveform(approximant=self.waveform_approximant,
                         mass1=self.x[ii,0],
                         mass2=self.x[ii,1],
                         spin1z=s1z,
                         spin2z=s2z,
                         distance=self.distance,
                         f_lower=self.f_low,
                         delta_f=self.delta_f)
            hptilde.resize(self.flen)
            # Generate the aLIGO ZDHP PSD
            psd = aLIGOZeroDetHighPower(self.flen, self.delta_f, self.f_low)
            mp, i = filter.match(segment, hptilde, psd=psd, low_frequency_cutoff=self.f_low)
            corr_mat[1,0]=mp
#print corr_mat
            wa, va = np.linalg.eig(corr_mat)
            drxn1 = np.argmax(wa)                   #larger eigenvalue
            drxn2 = np.argmin(wa)                   # Smaller eigenvalue
            self.x[ii,2:4]=self.x[ii,2:4]+np.dot(ds,va[:,drxn1])*va[:,drxn1]+np.dot(ds,va[:,drxn2])*va[:,drxn2]
#           self.v[ii,2:4]=self.v[ii,2:4]+np.dot(dv,va[:,drxn1])*va[:,drxn1]+np.dot(dv,va[:,drxn2])*va[:,drxn2]
#           ev=wa[drxn]*va[:,drxn]
#           self.x[ii,2]=x0[ii,2]-(ds1*va[0,0]+ds2*va[0,1])
#           self.x[ii,3]=x0[ii,3]-(ds1*va[1,0]+ds2*va[1,1])
#           self.v[ii,2]=v0[ii,2]-(dv1*va[0,1]+dv2*va[0,1])
#           self.v[ii,3]=v0[ii,3]-(dv1*va[1,1]+dv2*va[1,1])
        self.apply_reflecting_boundary()
#print wa
#print wa[drxn]*va[:,drxn]
###########################################################################
    def set_alpha(self, a) :
        '''
            changes the value of parameter alpha
        '''
        self.alpha=a
## end of set_alpha
##########################################################################
    def set_beta(self, b) :
        '''
            changes the value of parameter beta
        '''
        self.beta=b
## end of set_beta
##########################################################################
    def set_gamma(self, g) :
        '''
            changes the value of parameter gamma
        '''
        self.gamma=g
## end of set_gamma
##########################################################################
    #def print_pos_variables(self, fid=sys.stdout) :
        '''
             This is to print the complete state of PSO 
        '''
     #   print >> fid, "function: swarm.print_pos_variables"
     #   print >> fid, "Not implemented yet"
     #   pass
##############################################################################
    def generate_template_strain_freqdomain(self, Pjj,row):
        ifo=row.ifo
        d=Detector(ifo)
        pol=np.random.uniform(0,2*np.pi)
        ra, dec= d.optimal_orientation(self.trigger_time)
        Fp, Fc = d.antenna_pattern(ra,dec,pol,self.trigger_time)
        dim=self.dim
        if dim ==2 :
            hp0, hc0 = get_fd_waveform(approximant=self.waveform_approximant,
                    mass1=self.x[Pjj,0],
                    mass2=self.x[Pjj,1],
                    f_lower=self.f_low,
                    coa_phase=0.0,
                    distance=self.distance,
                    delta_f=self.delta_f)
            hp1, hc1 = get_fd_waveform(approximant=self.waveform_approximant,
                    mass1=self.x[Pjj,0],
                    mass2=self.x[Pjj,1],
                    f_lower=self.f_low,
                    coa_phase=np.pi/2,
                    distance=self.distance,
                    delta_f=self.delta_f)
           
        if dim ==3 :
            hp0, hc0 = get_fd_waveform(approximant=self.waveform_approximant,
                            mass1=self.x[Pjj,0],
                            mass2=self.x[Pjj,1],
                            spin1z=self.x[Pjj,2],
                            f_lower=self.f_low,
                            coa_phase=0.0,
                            distance=self.distance,
                            delta_f=self.delta_f)
            hp1, hc1 = get_fd_waveform(approximant=self.waveform_approximant,
                            mass1=self.x[Pjj,0],
                            mass2=self.x[Pjj,1],
                            spin1z=self.x[Pjj,2],
                            f_lower=self.f_low,
                            coa_phase=np.pi/2,
                            distance=self.distance,
                            delta_f=self.delta_f)
            
        if dim ==4 :
            hp0, hc0 = get_fd_waveform(approximant=self.waveform_approximant,
                            mass1=self.x[Pjj,0],
                            mass2=self.x[Pjj,1],
                            spin1z=self.x[Pjj,2],
                            spin2z=self.x[Pjj,3],
                            f_lower=self.f_low,
                            coa_phase=0.0,
                            distance=self.distance,
                            delta_f=self.delta_f)
            hp1, hc1 = get_fd_waveform(approximant=self.waveform_approximant,
                            mass1=self.x[Pjj,0],
                            mass2=self.x[Pjj,1],
                            spin1z=self.x[Pjj,2],
                            spin2z=self.x[Pjj,3],
                            f_lower=self.f_low,
                            coa_phase=np.pi/2,
                            distance=self.distance,
                            delta_f=self.delta_f) 
        hp0.resize(self.flen)
        hc0.resize(self.flen)
        hp1.resize(self.flen)
        hc1.resize(self.flen)
        ht0=hp0*Fp+hc0*Fc
        ht1=hp1*Fp+hc1*Fc
        return ht0, ht1
##############################################################################
    
##############################################################################
## End of class
##############################################################################

##############################################################################
## Some useful plotting function may be added here, as it is getting complecated 
## to write them in differrent code 

def plot_mean_distance(sw_list, file_name ):
    '''
        sw_list should be list of swarms
        file_name is where figure saved
    '''
    n_swarms=len(sw_list)
    s_dim=sw_list[0].dim
    plt.figure(figsize=(8, 6))
    for kk in range(n_swarms) :
        plt.plot(sw_list[kk].radius, '-ok',markerfacecolor='r')
    plt.xlabel('Steps')
    plt.ylabel('Mean Radius')
    plt.title('Swarm size for '+str(n_swarms)+' Number of swarms')
    plt.grid()
    plt.savefig(file_name)
    plt.close()

##############################################################################
def plot_swarm_gbest_snr(sw_list, file_name ):
    '''
        sw_list should be list of swarms
        file_name is where figure saved
    '''
    n_swarms=len(sw_list)
    s_dim=sw_list[0].dim
    plt.figure(figsize=(8, 6))
    for kk in range(n_swarms) :
        plt.plot(sw_list[kk].gbest_snr_step, '-ok',markerfacecolor='r')
    plt.xlabel('Steps')
    plt.ylabel('Gbest')
    plt.title('Swarm size for '+str(n_swarms)+' Number of swarms')
    plt.grid()
    plt.savefig(file_name)
    plt.close()

##############################################################################
def save_swarm_poistion(sw_list, file_name ):
    try :
        fd=open(file_name,'wv')
        pickle.dump(sw_list,fd)
        fd.close()
    except OSError :
        print('unable to write to data file '+file_name)
##############################################################################
def plot_swarm_poistion(sw_list, file_name ):
    search_dim=sw_list[0].dim
    n_swarms=len(sw_list)
    if search_dim==2 :
        plt.figure()
    if search_dim==3 :
        plt.figure(figsize=(18,6))
    if search_dim==4 :
        plt.figure(figsize=(12,18))
    xx=np.linspace(sw_list[0].min[0], sw_list[0].max[0], 100)
    plt.title('PSO step '+str(sw_list[0].curr_step).rjust(4,'0'))
    
    for kk in range(n_swarms):
        if search_dim==2 :
            plt.plot(sw_list[kk].x[:,0], sw_list[kk].x[:,1],'bo', markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[0], sw_list[kk].gbest[1], '*r')
            plt.quiver(sw_list[kk].x[:,0],sw_list[kk].x[:,1],sw_list[kk].v[:,0], sw_list[kk].v[:,1])
        if search_dim==3 :
            plt.subplot(1,3,1)
            plt.plot(sw_list[kk].x[:,0], sw_list[kk].x[:,1],'bo', markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[0], sw_list[kk].gbest[1], '*r')
            plt.quiver(sw_list[kk].x[:,0],sw_list[kk].x[:,1],sw_list[kk].v[:,0], sw_list[kk].v[:,1])
            plt.subplot(1,3,2)
            plt.plot(sw_list[kk].x[:,0], sw_list[kk].x[:,2],'bo',markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[0], sw_list[kk].gbest[2], '*r')
            plt.quiver(sw_list[kk].x[:,0],sw_list[kk].x[:,2],sw_list[kk].v[:,0], sw_list[kk].v[:,2])       
            plt.subplot(1,3,3)
            plt.plot(sw_list[kk].x[:,1], sw_list[kk].x[:,2],'bo',markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[1], sw_list[kk].gbest[2], '*r')
            plt.quiver(sw_list[kk].x[:,1],sw_list[kk].x[:,2],sw_list[kk].v[:,1], sw_list[kk].v[:,2])
        if search_dim==4 :
            plt.subplot(3,2,1)
            plt.plot(sw_list[kk].x[:,0], sw_list[kk].x[:,1],'bo', markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[0], sw_list[kk].gbest[1], '*r')
            plt.quiver(sw_list[kk].x[:,0],sw_list[kk].x[:,1],sw_list[kk].v[:,0], sw_list[kk].v[:,1])
            plt.subplot(3,2,3)
            plt.plot(sw_list[kk].x[:,0], sw_list[kk].x[:,2],'bo',markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[0], sw_list[kk].gbest[2], '*r')
            plt.quiver(sw_list[kk].x[:,0],sw_list[kk].x[:,2],sw_list[kk].v[:,0], sw_list[kk].v[:,2])
            plt.subplot(3,2,4)
            plt.plot(sw_list[kk].x[:,0], sw_list[kk].x[:,3],'bo',markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[0], sw_list[kk].gbest[3], '*r')
            plt.quiver(sw_list[kk].x[:,0],sw_list[kk].x[:,3],sw_list[kk].v[:,0], sw_list[kk].v[:,3])
            plt.subplot(3,2,5)
            plt.plot(sw_list[kk].x[:,1], sw_list[kk].x[:,2],'bo',markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[1], sw_list[kk].gbest[2], '*r')
            plt.quiver(sw_list[kk].x[:,1],sw_list[kk].x[:,2],sw_list[kk].v[:,1], sw_list[kk].v[:,2])           
            plt.subplot(3,2,6)
            plt.plot(sw_list[kk].x[:,1], sw_list[kk].x[:,3],'bo',markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[1], sw_list[kk].gbest[3], '*r')
            plt.quiver(sw_list[kk].x[:,1],sw_list[kk].x[:,3],sw_list[kk].v[:,1], sw_list[kk].v[:,3])           
            plt.subplot(3,2,2)
            plt.plot(sw_list[kk].x[:,2], sw_list[kk].x[:,3],'bo', markerfacecolor='w')
            plt.plot(sw_list[kk].gbest[2], sw_list[kk].gbest[3], '*r')
            plt.quiver(sw_list[kk].x[:,2],sw_list[kk].x[:,3],sw_list[kk].v[:,2], sw_list[kk].v[:,3])
    if search_dim==2 :
        plt.plot(xx, xx, 'g')
        plt.plot(sw_list[0].sim_insp_par.mass1, sw_list[0].sim_insp_par.mass2, 'cD' )
        plt.grid()
        plt.xlim([sw_list[0].min[0], sw_list[0].max[0]])
        plt.ylim([sw_list[0].min[1], sw_list[0].max[1]])
        plt.xlabel(sw_list[0].parameters_names[0])
        plt.ylabel(sw_list[0].parameters_names[1])
        plt.title(sw_list[0].parameters_names[0]+' And '+sw_list[0].parameters_names[1]+'Plane'+'  '+str(sw_list[0].curr_step))
    if search_dim==3 :
        plt.subplot(1,3,1)
        plt.plot(xx, xx, 'g')
        plt.title(sw_list[0].parameters_names[0]+' And '+sw_list[0].parameters_names[1]+'Plane')
        plt.xlim([sw_list[kk].min[0], sw_list[kk].max[0]])
        plt.ylim([sw_list[kk].min[1], sw_list[kk].max[1]])
        plt.xlabel(sw_list[0].parameters_names[0])
        plt.ylabel(sw_list[0].parameters_names[1])
        plt.grid()
        plt.subplot(1,3,2)
        plt.title(sw_list[0].parameters_names[0]+' And '+sw_list[0].parameters_names[2]+'Plane')
        plt.xlim([sw_list[kk].min[0], sw_list[kk].max[0]])
        plt.ylim([sw_list[kk].min[2], sw_list[kk].max[2]])    
        plt.xlabel(sw_list[0].parameters_names[0])
        plt.ylabel(sw_list[0].parameters_names[2])
        plt.grid()
        plt.subplot(1,3,3)
        plt.grid()
        plt.title(sw_list[0].parameters_names[1]+' And '+sw_list[0].parameters_names[2]+'Plane')
        plt.xlim([sw_list[kk].min[1], sw_list[kk].max[1]])
        plt.ylim([sw_list[kk].min[2], sw_list[kk].max[2]])
        plt.xlabel(sw_list[0].parameters_names[1])
        plt.ylabel(sw_list[0].parameters_names[2])
        plt.grid()
    if search_dim==4 :
        plt.subplot(3,2,1)
        plt.plot(xx, xx, 'g')
        plt.title(sw_list[0].parameters_names[0]+' And '+sw_list[0].parameters_names[1]+'Plane')
        plt.xlim([sw_list[kk].min[0], sw_list[kk].max[0]])
        plt.ylim([sw_list[kk].min[1], sw_list[kk].max[1]])
        plt.xlabel(sw_list[0].parameters_names[0])
        plt.ylabel(sw_list[0].parameters_names[1])
        plt.grid()
        plt.subplot(3,2,3)
        plt.title(sw_list[0].parameters_names[0]+' And '+sw_list[0].parameters_names[2]+'Plane')
        plt.xlim([sw_list[kk].min[0], sw_list[kk].max[0]])
        plt.ylim([sw_list[kk].min[2], sw_list[kk].max[2]])
        plt.xlabel(sw_list[0].parameters_names[0])
        plt.ylabel(sw_list[0].parameters_names[2])
        plt.grid()
        plt.subplot(3,2,4)
        plt.title(sw_list[0].parameters_names[0]+' And '+sw_list[0].parameters_names[3]+'Plane')
        plt.xlim([sw_list[kk].min[0], sw_list[kk].max[0]])
        plt.ylim([sw_list[kk].min[3], sw_list[kk].max[3]])
        plt.xlabel(sw_list[0].parameters_names[0])
        plt.ylabel(sw_list[0].parameters_names[3])
        plt.grid()
        plt.subplot(3,2,5)
        plt.title(sw_list[0].parameters_names[1]+' And '+sw_list[0].parameters_names[2]+'Plane')
        plt.xlim([sw_list[kk].min[1], sw_list[kk].max[1]])
        plt.ylim([sw_list[kk].min[2], sw_list[kk].max[2]])
        plt.xlabel(sw_list[0].parameters_names[1])
        plt.ylabel(sw_list[0].parameters_names[2])
        plt.grid()
        plt.subplot(3,2,6)
        plt.title(sw_list[0].parameters_names[1]+' And '+sw_list[0].parameters_names[3]+'Plane')
        plt.xlim([sw_list[kk].min[1], sw_list[kk].max[1]])
        plt.ylim([sw_list[kk].min[3], sw_list[kk].max[3]])
        plt.xlabel(sw_list[0].parameters_names[1])
        plt.ylabel(sw_list[0].parameters_names[3])
        plt.grid()
        plt.subplot(3,2,2)
        plt.title(sw_list[0].parameters_names[2]+' And '+sw_list[0].parameters_names[3]+'Plane')
        plt.xlim([sw_list[kk].min[2], sw_list[kk].max[2]])
        plt.ylim([sw_list[kk].min[3], sw_list[kk].max[3]])
        plt.xlabel(sw_list[0].parameters_names[2])
        plt.ylabel(sw_list[0].parameters_names[3])
        plt.grid()
    plt.savefig(file_name)
    plt.close()
##############################################################################
class double_swarm:
    '''
        This is two swarm class for evolving two swarms exchaning information and 
        optimising thier path
    '''
    def __init__(self,sps, tmax, sim_insp_par, ifo='L1',flow=20, particles=200, steps=30, dim=2):
        self.sw1=swarm(particles=particles, steps=particles, dim=dim)
        self.sw2=swarm(particles=particles, steps=particles, dim=dim)
        self.x10=np.zeros(self.sw1.x.shape)
        self.x20=np.zeros(self.sw1.x.shape)
        ## Why below variables are needed!
        self.corr00=np.zeros(self.sw1.Np)
        self.corr01=np.zeros(self.sw1.Np)
        #self.corr10=np.zeros(self.sw1.Np)
        #self.corr11=np.zeros(self.sw1.Np)
        self.curr_step=0
        self.corr_mat=np.zeros((2,2))
        self.gbest=np.zeros(self.dim)
        self.gbest_snr = 0.0
        
    def evolve_two_swarm_eigen_direction(self,segment, drxn=0 ) :
        '''
            This funtion veloves two swarm as standard PSO and moves along eigen direction
            segment -> signal segment
            drxn    -> 0 means, standard PSO do nothing
                    -> 1 means , {m1, m2} eigen direction
                    -> 3 means,  {s1z, s2z} eigen direction
                    -> 4 means both {m1, m2} and {s1z, s2z}
        '''
        
        self.sw1.compute_fitness(segment)
        self.sw1.compute_pbest()
        self.sw1.compute_gbest()
        self.sw2.compute_fitness(segment)
        self.sw2.compute_pbest()
        self.sw2.compute_gbest()
        self.x10[:]=self.sw1.x[:]
        self.x20[:]=self.sw2.x[:]
        self.corr00[:]=self.sw1.snr[:]
        self.corr01[:]=self.sw2.snr[:]
        self.sw1.evolve_velocity()
        self.sw1.evolve_position()
        self.sw1.apply_reflecting_boundary()
        self.sw2.evolve_velocity()
        self.sw2.evolve_position()
        self.sw2.apply_reflecting_boundary()
        self.sw1.compute_fitness(segment)
        self.sw2.compute_fitness(segment)
        Npmin=min(self.sw1.Np, self.sw2.Np)
        for ii in range(Npmin) :
            dx=self.sw1.x[ii,0:2]-self.sw2.x[ii,0:2]
            dv=self.sw1.v[ii,0:2]-self.sw2.v[ii,0:2]
            self.corr_mat[0,0]=self.corr00[ii]
            self.corr_mat[0,1]=self.corr01[ii]
            self.corr_mat[1,0]=self.sw1.snr[ii]
            self.corr_mat[1,1]=self.sw2.snr[ii]
            wa,ev=np.linalg.eig(self.corr_mat)
            drxn1 = np.argmax(wa)                   #larger eigenvalue
            drxn2 = np.argmin(wa)                   # Smaller eigenvalue
            self.sw1.x[ii,:]=self.x10[ii,:]+np.dot(dx,ev[:,drxn1])*ev[:,drxn1]
            self.sw2.x[ii,:]=self.x20[ii,:]+np.dot(dx,ev[:,drxn2])*ev[:,drxn2]
            #self.sw1.v[ii,:]=self.sw1.v[ii,:]+np.dot(dv,ev[:,drxn1])*ev[:,drxn1]
            #self.sw2.v[ii,:]=self.sw2.v[ii,:]+np.dot(dv,ev[:,drxn2])*ev[:,drxn2]
      
            
        self.sw1.apply_reflecting_boundary()
        self.sw2.apply_reflecting_boundary()
        self.sw1.compute_radius()
        self.sw2.compute_radius()
    def compute_gbest(self):
        s1=self.sw1.gbest_snr
        s2=self.sw2.gbest_snr
        if s1 > s2 :
            self.sw2.gbest_snr=s1
            self.sw2.gbest=self.sw1.gbest
            self.gbest_snr=s1
            self.gbest=self.sw1.gbest
        else :
            self.sw1.gbest_snr=s2
            self.sw1.gbest=self.sw2.gbest
            self.gbest_snr=s2
            self.gbest=self.sw1.gbest
        
            
            
        
##############################################################################
def generate_timedomain_strain(row):
    print('Phase from polarization')
    ifo=row.ifo
    search_dim=row.search_dim
    signal_sps=row.sps
    d=Detector(ifo)
    Fp, Fc = d.antenna_pattern(row.ra, row.dec, row.pol, row.trig_time)
    delta_f=1.0/row.tmax
    delta_t=1.0/4096
    tlen=int(row.tmax*row.sps)
    flen=int(tlen/2)+1
    print(ifo)
    if ifo == 'L1' or ifo == 'H1' or ifo=='I1' :
        psd=aLIGOZeroDetHighPower(flen, delta_f,row.f_lower ) 
    elif ifo == 'V1' :
        psd=AdvVirgo(flen, delta_f,row.f_lower)
    elif ifo == 'K1':
        psd=KAGRA(flen, delta_f,row.f_lower)
    elif ifo == 'L2' or ifo == 'H2' :    ## L2 and H2 stands for 02 noise PSD     
        psd = from_txt(row.O2_PSD_file,flen, delta_f, row.f_lower,False)
    else :
        raise Exception('Unknown IFO :'+self.ifo)
    wvform = row.waveform
    if row.signal_stat != 1 : 
        noise = pycbc.noise.gaussian.noise_from_psd(tlen, delta_t, psd, seed=row.noise_seed)
        #noise.resize(tlen)
    if row.signal_stat == 3 or row.signal_stat == 4 :
        ts=frame.read_frame(row.frame_file_name,row.frame_ch)
        ts = highpass(ts, row.f_lower)
        ts = lowpass_fir(ts, 1200, 8)
        t0=int((row.trig_time-row.st_time)*row.sps)
        t1=t0+int(row.tmax*row.sps)
        ts=ts[t0:t1]
        ts= ts.detrend('constant')
        #mx=2*max(ts)
        #y=np.linspace(-mx, mx, 30)
        plt.plot(ts.sample_times, ts,'r')
        plt.savefig('plots/00_detector_output_time_series.png')
        #plt.hist(ts)
        #plt.savefig('plots/00_det_output_hist.png')
        noise=ts.to_frequencyseries(delta_f=delta_f) ####????
    if (row.signal_stat != 0 or row.signal_stat != 3) and search_dim ==2 :
        hp, hc = get_td_waveform(approximant=row.waveform,mass1=row.mass1,mass2=row.mass2,f_lower=row.f_lower,coa_phase=0.0,distance=row.distance,delta_t=delta_t)
        hp.resize(tlen)
        hc.resize(tlen)
    if (row.signal_stat != 0 or row.signal_stat != 3) and search_dim ==3 :
        hp, hc = get_td_waveform(approximant=row.waveform,
                mass1=row.mass1,
                mass2=row.mass2,
                spin1z=row.spin1z,
                f_lower=row.f_lower,
                coa_phase=0.0,
                distance=row.distance,
                delta_t=delta_t)
        hp.resize(tlen)
        hc.resize(tlen)
    if (row.signal_stat != 0 or row.signal_stat != 3) and search_dim ==4 :
        hp, hc = get_td_waveform(approximant=row.waveform,
                mass1=row.mass1,
                mass2=row.mass2,
                spin1z=row.spin1z,
                spin2z=row.spin2z,
                f_lower=row.f_lower,
                coa_phase=0.0,
                distance=row.distance,
                delta_t=delta_t)  
        hp.resize(tlen)
        hp.resize(tlen)
    if row.signal_stat == 1 :
        
        signal = hp*Fp+hc*Fc
    elif row.signal_stat == 0 or row.signal_stat == 3:
        signal = noise
    else :
        print(len(hp), len(noise))
        sig=hp*Fp+hc*Fc
        sig.start_time=0
        signal = sig+noise 
        
    #phase=waveform.utils.phase_from_polarizations(hp, hc)
    #sig=fft(sig)
    return signal, psd
##############################################################################
def generate_freqdomain_strain(row):
    ifo=row.ifo
    search_dim=row.search_dim
    signal_sps=row.sps
    delta_f=1.0/row.tmax
    tlen=int(row.tmax*row.sps)
    flen=int(tlen/2)+1
    #print('flen=', flen)
    print(ifo)
    d=Detector(ifo)
    Fp, Fc = d.antenna_pattern(row.ra, row.dec, row.pol, row.trig_time)
    if ifo == 'L1' or ifo == 'H1' or ifo=='I1' :
        psd=aLIGOZeroDetHighPower(flen, delta_f,row.f_lower ) 
    elif ifo == 'V1' :
        psd=AdvVirgo(flen, delta_f,row.f_lower)
    elif ifo == 'K1':
        psd=KAGRA(flen, delta_f,row.f_lower)
    elif ifo == 'L2' or ifo == 'H2' :    ## L2 and H2 stands for 02 noise PSD     
        psd = from_txt(row.O2_PSD_file,flen, delta_f, row.f_lower,False)
    else :
        raise Exception('Unknown IFO :'+self.ifo)
    wvform = row.waveform
    if row.signal_stat != 1 : 
        noise = frequency_noise_from_psd(psd, seed=row.noise_seed)
    if row.signal_stat == 3 or row.signal_stat == 4 :
        ts=frame.read_frame(row.frame_file_name,row.frame_ch)
        ts = highpass(ts, row.f_lower)
        ts = lowpass_fir(ts, 1200, 8)
        t0=int((row.trig_time-row.st_time)*row.sps)
        t1=t0+int(row.tmax*row.sps)
        ts=ts[t0:t1]
        ts= ts.detrend('constant')
        #mx=2*max(ts)
        #y=np.linspace(-mx, mx, 30)
        plt.plot(ts.sample_times, ts,'r')
        plt.savefig('plots/00_detector_output_time_series.png')
        #plt.hist(ts)
        #plt.savefig('plots/00_det_output_hist.png')
        noise=ts.to_frequencyseries(delta_f=delta_f)
                                        
        
    if (row.signal_stat != 0 or row.signal_stat != 3) and search_dim ==2 :
        hp, hc = get_fd_waveform(approximant=wvform,
                mass1=row.mass1,
                mass2=row.mass2,
                f_lower=row.f_lower,
                coa_phase=0.0,
                distance=row.distance,
                delta_f=delta_f)
        hp.resize(flen)
        hc.resize(flen)
    if (row.signal_stat != 0 or row.signal_stat != 3) and search_dim ==3 :
        hp, hc = get_fd_waveform(approximant=wvform,
                mass1=row.mass1,
                mass2=row.mass2,
                spin1z=row.spin1z,
                f_lower=row.f_lower,
                coa_phase=0.0,
                distance=row.distance,
                delta_f=delta_f)
        hp.resize(flen)
        hc.resize(flen)
    if (row.signal_stat != 0 or row.signal_stat != 3) and search_dim ==4 :
        hp, hc = get_fd_waveform(approximant=wvform,
                mass1=row.mass1,
                mass2=row.mass2,
                spin1z=row.spin1z,
                spin2z=row.spin2z,
                f_lower=row.f_lower,
                coa_phase=0.0,
                distance=row.distance,
                delta_f=delta_f)  
        hp.resize(flen)
        hp.resize(flen)
    if row.signal_stat == 1 :
        
        sig = hp*Fp+hc*Fc
    elif row.signal_stat == 0 or row.signal_stat == 3:
        sig = noise
    else :
        #print(len(hp), len(noise))
        sig = hp*Fp+hc*Fc+noise 
    #p=waveform.utils.phase_from_frequencyseries(sig, remove_start_phase=True)	
    
    return sig, psd    
##############################################################################
'''def dataset(self,steps,particles,chirp_mass,match):
    ''''''Creating Dataset for training the neural network''''''
    steps=steps
    paticles=particles
    
    with open('NN_Dataset.csv', mode='w') as csv_file:
       fieldnames=['particle', 'step','chirp_mass', 'match', 'Target']
       writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
       
       writer.writerheader()
       for ii in range(particles):
               for jj in range(steps):
               	writer.writerow({'particle': ii, 'step': jj, 'chirp_mass' = chrip_mass[ii,jj], 'match' = match[ii,jj]})
 '''              
##############################################################################
def creat_workin_directory(sim_id, over_write):
    try :
        #print('Creating directory '+sim_id)
        os.mkdir(sim_id)
        ## creat director
    except OSError :
        ## It exists ? Remove it and create anyway
        print(sim_id+' Exists')
        if over_write=='y' :
            print('Over writing')
            shutil.rmtree(sim_id)
            os.mkdir(sim_id)
        else :
            sim_id=''
    return sim_id
##############################################################################
class pso_log :
    fd=None 
    def __init__(self, file_name=None) :
        if file_name != None :
            self.fd=open(file_name,'wt')
            print('#'*20,file=self.fd) 
            print('Log file created',file=self.fd) 
            print(datetime.now().ctime(),file=self.fd) 
            print('#'*20,file=self.fd) 
            self.fd.flush()
        elif self.fd == None :
            raise Exception('Log file not created!')
    def write(self, bluf) :
        try :
            print( bluf,file=self.fd) 
            self.fd.flush()
        except OSError  :
            print('Error writing log')
        
    def __del__(self) :
        self.fd.close()
            
##############################################################################    
if __name__ == "__main__":
    print('Module only, no main')
##############################################################################
## Project contributers 
## Anuradha Samajdar
## Varun Srivastava
## Shubhagata Bhaumik
## Ankit Mandal
## Tathagata Pal
## Aritra Aich
## Souradeep Pal
## Gopi Patel
## Rajesh Kumble Nayak
## Sukanta Bose
