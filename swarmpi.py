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
from scipy import fft
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
import glob
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
from astropy.coordinates import SkyCoord
from astropy import units as u
import seaborn as sns
from multiprocessing import Pool

##
##
###############################################################################
## Class swarm
## definition
###############################################################################
class inj_par :
    m1min=15.0    ## Ms
    m1max=40.0    ## Ms
    m2min=15.0    ## Ms
    m2max=40.0    ## Ms
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
        waveform='IMRPhenomC'  -> trivial
        snr              ->   function value at x,  Np array
        radius            ->   mean distance from g_best  array of size steps
        DEBUG             ->   can be False or True
    '''
##########################################################################
    def __init__(self,sps, tmax, ifo='I1',flow=20, particles=200, steps=30, dim=2, waveform='IMRPhenomC'):
        '''


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
        self.curr_step=0
        self.tlen=int(self.sps*self.T_max)
        self.flen=int(self.tlen/2)+1
        self.alpha=0.4
        self.beta=0.3
        self.gamma=0.35
        self.distance=1.0                  ## MPC
        self.trigger_time=0.0
        self.waveform=waveform
        self.ifo=ifo
        self.norm=1.0
        self.DEBUG = False
        self.run_pll=False
        self.nof_process=5
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
        self.pbest_phs=np.zeros(self.Np)
        self.gbest=np.zeros(self.dim)
        self.gbest_snr = 0.0#variable to store best match
        self.gbest_toa = 0.0
        self.gbest_phs = 0.0
        self.snr = np.zeros(self.Np)
        self.toa = np.zeros(self.Np)
        self.phs = np.zeros(self.Np)
        self.radius=np.zeros(self.Nsteps)
        self.gbest_snr_step=np.zeros(self.Nsteps)
        self.gbest_toa_step=np.zeros(self.Nsteps)
        self.gbest_phs_step=np.zeros(self.Nsteps)
        #self.sim_insp_par=sim_insp_par
        self.evolve=self.evolve_standard_pso
        self.evolve_option=0
        if self.ifo == 'L1' or self.ifo == 'H1' or self.ifo == 'I1' :
            self.psd = aLIGOZeroDetHighPower(self.flen, self.delta_f, self.f_low)
        elif self.ifo == 'V1' :
            self.psd = AdvVirgo(self.flen, self.delta_f, self.f_low)
        elif self.ifo == 'K1' :
            self.psd = KAGRA(self.flen, self.delta_f, self.f_low)
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
        del self.trigger_time, self.waveform, self.ifo
        del self.gbest_snr_step
##########################################################################
    def compute_fitness(self, segment):
        '''
            This is the function which is optmised!
        '''
        if not self.run_pll :
            for jj in range (self.Np):#Correlation foreach point
                temp0, temp1=self.generate_template_strain_freqdomain(jj)
                csnr0=pycbc.filter.matched_filter(temp0, segment, psd=self.psd,  low_frequency_cutoff=self.f_low,high_frequency_cutoff=self.f_upper)
                csnr1=pycbc.filter.matched_filter(temp1,segment , psd=self.psd,  low_frequency_cutoff=self.f_low,high_frequency_cutoff=self.f_upper)
                snr0,idx0=csnr0.abs_max_loc()
                snr1,idx1=csnr1.abs_max_loc()
                #csnr0=csnr0*np.conj(csnr0)
                #csnr1=csnr1*np.conj(csnr1)
                #idx0=np.argmax(csnr0)
                #idx1=np.argmax(csnr1)
                #snr0=csnr0[idx0]
                #snr1=csnr1[idx1]
                mp=0.5*np.sqrt(snr0**2+snr1**2)
                self.snr[jj] = np.real(mp) # v1_norm=None, v2_norm=None
                self.toa[jj] = idx0
                phase = 2*np.pi - np.angle(csnr0[idx0])
                self.phs[jj] = phase % (2*np.pi)
        else :
            p_pool=Pool(self.nof_process)
            res=[]
            for jj in range (0,self.Np,self.nof_process):#Correlation foreach point
                for kk in range(jj, min(jj+self.nof_process, self.Np)) :
                    res.append(p_pool.apply_async( self.generate_template_strain_freqdomain, (kk,)))
            kk=0
            flag=True
            while flag:
                try :
                    pp=res.pop(0)
                    temp0, temp1=pp.get()
                    csnr0=pycbc.filter.matched_filter(temp0,
                        segment,
                        psd=self.psd,
                        low_frequency_cutoff=self.f_low,
                        high_frequency_cutoff=self.f_upper)
                    csnr1=pycbc.filter.matched_filter(temp1,segment,
                        psd=self.psd,
                        low_frequency_cutoff=self.f_low,
                        high_frequency_cutoff=self.f_upper)
                    csnr0=csnr0*np.conj(csnr0)
                    csnr1=csnr1*np.conj(csnr1)
                    idx0=np.argmax(csnr0)
                    idx1=np.argmax(csnr1)
                    snr0=csnr0[idx0]
                    snr1=csnr1[idx1]
                    mp=0.5*np.sqrt(snr0+snr1)
                    self.snr[jj+kk] = np.real(mp) # v1_norm=None, v2_norm=None
                    self.toa[jj+kk] =idx0
                    kk+=1
                except IndexError :
                    flag=False

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
                self.pbest_phs[jj]=self.phs[jj]
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
            self.gbest_phs=self.pbest_phs[idbest]
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
            self.x[jj,:]=self.x[jj,:]+self.v[jj,:]
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
        self.gbest_phs_step[self.curr_step]=self.gbest_phs
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
            for jj in idx :
                tmp=self.x[jj,0]
                tmpv=self.v[jj,0]
                self.x[jj,0]=self.x[jj,1]
                self.v[jj,0]=self.v[jj,1]
                self.x[jj,1]=tmp
                self.v[jj,1]=tmpv
            #self.x[idx,1]=2*self.x[idx,0]-self.x[idx,1]
            #self.v[idx,0:2]=-self.v[idx,0:2]
            #self.apply_velocity_condition()
            for jj in range(2):
                idx=np.where(self.x[:,jj] <= 0)
                self.x[idx,jj]=self.min[jj]
                self.v[idx,jj]=-self.v[idx,jj]



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
    def evolve_standard_pso(self, segment) :
        '''
            This function takes all it needed to take one step as per standard
            PSO
        '''
        #self.compute_radius()
        self.compute_fitness(segment)
        self.compute_pbest()
        self.compute_gbest()
        self.store_gbest()
        self.evolve_velocity()
        self.evolve_position()
        self.apply_reflecting_boundary()
## end of  evolve_standard_pso
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
            hptilde, hctilde = get_fd_waveform(approximant=self.waveform,
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
            hptilde, hctilde = get_fd_waveform(approximant=self.waveform,
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

            hptilde, hctilde = get_fd_waveform(approximant=self.waveform,
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
            hptilde, hctilde = get_fd_waveform(approximant=self.waveform,
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
    def generate_template_strain_freqdomain(self, Pjj):
        ifo=self.ifo
        d=Detector(ifo)
        pol=np.random.uniform(0,2*np.pi)
        ra, dec= d.optimal_orientation(self.trigger_time)
        Fp, Fc = d.antenna_pattern(ra,dec,pol,self.trigger_time)
        dim=self.dim
        if dim ==2 :
            hp0, hc0 = get_fd_waveform(approximant=self.waveform,
                    mass1=self.x[Pjj,0],
                    mass2=self.x[Pjj,1],
                    f_lower=self.f_low,
                    coa_phase=0.0,
                    distance=self.distance,
                    delta_f=self.delta_f)
            hp1, hc1 = get_fd_waveform(approximant=self.waveform,
                    mass1=self.x[Pjj,0],
                    mass2=self.x[Pjj,1],
                    f_lower=self.f_low,
                    coa_phase=np.pi/2,
                    distance=self.distance,
                    delta_f=self.delta_f)

        if dim ==3 :
            hp0, hc0 = get_fd_waveform(approximant=self.waveform,
                            mass1=self.x[Pjj,0],
                            mass2=self.x[Pjj,1],
                            spin1z=self.x[Pjj,2],
                            f_lower=self.f_low,
                            coa_phase=0.0,
                            distance=self.distance,
                            delta_f=self.delta_f)
            hp1, hc1 = get_fd_waveform(approximant=self.waveform,
                            mass1=self.x[Pjj,0],
                            mass2=self.x[Pjj,1],
                            spin1z=self.x[Pjj,2],
                            f_lower=self.f_low,
                            coa_phase=np.pi/2,
                            distance=self.distance,
                            delta_f=self.delta_f)

        if dim ==4 :
            hp0, hc0 = get_fd_waveform(approximant=self.waveform,
                            mass1=self.x[Pjj,0],
                            mass2=self.x[Pjj,1],
                            spin1z=self.x[Pjj,2],
                            spin2z=self.x[Pjj,3],
                            f_lower=self.f_low,
                            coa_phase=0.0,
                            distance=self.distance,
                            delta_f=self.delta_f)
            hp1, hc1 = get_fd_waveform(approximant=self.waveform,
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
        #ht0=hp0*Fp+hc0*Fc
        #ht1=hp1*Fp+hc1*Fc
        ht0=hp0             # Let us test with this
        ht1=hp1
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
        fd=open(file_name,'w')
        pickle.dump(sw_list,fd)
        fd.close()
    except OSError :
        print('unable to write to data file '+file_name)
##############################################################################
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
#####################################################################################################
def generate_data_segment(row):
    ifo=row.ifo
    d=Detector(ifo)
    signal_sps=row.sps
    delta_f=1.0/row.tmax
    delta_t=1.0/row.sps
    tlen=int(row.tmax*row.sps)
    flen=int(tlen/2)+1
    wvform = row.waveform
    trig_time=row.geocent_end_time
    st_time=row.st_time
    if row.injx_mode== 0 or row.injx_mode== 2 :
        if ifo == 'L1' or ifo == 'H1' or ifo=='I1' :
            psd=aLIGOZeroDetHighPower(flen, delta_f,row.f_lower )
        elif ifo == 'V1' :
            psd=AdvVirgo(flen, delta_f,row.f_lower)
        elif ifo == 'K1':
            psd=KAGRA(flen, delta_f,row.f_lower)
        else :
            raise Exception('Unknown IFO :'+self.ifo)
        ##print('noise seed=', row.noise_seed)
        noise = pycbc.noise.gaussian.noise_from_psd( tlen,  delta_t, psd, seed=row.noise_seed)
        #print(len(noise))
        if row.injx_mode== 0 :
            signal = noise
        #noise.resize(tlen)

    if row.injx_mode == 1 or row.injx_mode == 2 :
        hp, hc = get_td_waveform(approximant=row.waveform,
                mass1=row.mass1,
                mass2=row.mass2,
                spin1x=row.spin1x,
                spin2x=row.spin2x,
                spin1y=row.spin1y,
                spin2y=row.spin2y,
                spin1z=row.spin1z,
                spin2z=row.spin2z,
                f_lower=row.f_lower,
                coa_phase=row.coa_phase,
                inclination=row.inclination,
                polarization=row.polarization,
                distance=row.distance,
                delta_t=delta_t)
        noise.start_time=st_time
        hp.start_time=trig_time
        hc.start_time=trig_time
        signal = d.project_wave( hp, hc, row.ra_dec[0], row.ra_dec[1], row.polarization, method='constant', reference_time=trig_time )
        if False :
            plt.plot(noise.sample_times, noise, 'b', label='Noise')
            plt.plot(signal.sample_times, signal, 'r', label='Signal')
            plt.legend()
            plt.savefig('signal.png')
            exit(-1)
        signal=noise.inject(signal)
        if False  :
            plt.plot(signal.sample_times, signal, 'C0', label='Signal')
            plt.legend()
            plt.savefig('signal.png')
            exit(-1)
    return signal, psd
##############################################################################
def generate_timedomain_strain(row):    ## would be depricate soon
    ifo=row.ifo
    search_dim=row.search_dim
    d=Detector(ifo)
    if hasattr(row, 'row') :
        row1=row.row
        Fp, Fc = d.antenna_pattern(row1.ra_dec[0], row1.ra_dec[1], row1.polarization, row1.time_geocent)
    else :
        row1=row
        Fp, Fc = d.antenna_pattern(row1.ra, row1.dec, row1.pol, row1.trig_time)
    signal_sps=row.sps
    #Fp, Fc = d.antenna_pattern(row1.ra_dec[0], row1.ra_dec[1], row1.polarization, row1.trig_time)
    print("row.tmax",row.tmax)
    delta_f=1.0/row.tmax
    delta_t=1.0/4096
    tlen=int(row.tmax*row.sps)
    flen=int(tlen/2)+1
    print(ifo)
    if ifo == 'L1' or ifo == 'H1' or ifo=='I1' :
        psd=aLIGOZeroDetHighPower(flen, delta_f,row1.f_lower )
    elif ifo == 'V1' :
        psd=AdvVirgo(flen, delta_f,row1.f_lower)
    elif ifo == 'K1':
        psd=KAGRA(flen, delta_f,row1.f_lower)
    elif ifo == 'L2' or ifo == 'H2' :    ## L2 and H2 stands for 02 noise PSD
        psd = from_txt(row.O2_PSD_file,flen, delta_f, row1.f_lower,False)
    else :
        raise Exception('Unknown IFO :'+self.ifo)
    wvform = row1.waveform
    if row.signal_stat != 1 :
        noise = pycbc.noise.gaussian.noise_from_psd(tlen, delta_t, psd, seed=row.noise_seed)
        #noise.resize(tlen)
    if row.signal_stat == 3 or row.signal_stat == 4 :
        ts=frame.read_frame(row.frame_file_name,row.frame_ch)
        ts = highpass(ts, row1.f_lower)
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
        hp, hc = get_td_waveform(approximant=row1.waveform,mass1=row1.mass1,mass2=row1.mass2,f_lower=row1.f_lower,coa_phase=0.0,distance=row1.distance,delta_t=delta_t)
        hp.resize(tlen)
        hc.resize(tlen)
    if (row.signal_stat != 0 or row.signal_stat != 3) and search_dim ==3 :
        hp, hc = get_td_waveform(approximant=row1.waveform,
                mass1=row1.mass1,
                mass2=row1.mass2,
                spin1z=row1.spin1z,
                f_lower=row1.f_lower,
                coa_phase=0.0,
                distance=row1.distance,
                delta_t=delta_t)
        hp.resize(tlen)
        hc.resize(tlen)
    if (row.signal_stat != 0 or row.signal_stat != 3) and search_dim ==4 :
        hp, hc = get_td_waveform(approximant=row1.waveform,
                mass1=row1.mass1,
                mass2=row1.mass2,
                spin1z=row1.spin1z,
                spin2z=row1.spin2z,
                f_lower=row1.f_lower,
                coa_phase=0.0,
                distance=row1.distance,
                delta_t=delta_t)
        hp.resize(tlen)
        hp.resize(tlen)
    if row.signal_stat == 1 :

        sig = hp*Fp+hc*Fc
    elif row.signal_stat == 0 or row.signal_stat == 3:
        sig = noise
    else :
        sig=hp*Fp+hc*Fc
        sig.start_time=0
        signal = sig+noise

    #phase=waveform.utils.phase_from_polarizations(hp, hc)
    #sig=fft(sig)
    plt.plot(noise.sample_times, noise, 'b', label='noise')
    plt.plot(sig.sample_times, sig, 'r', label='signal')
    plt.legend()
    plt.xlabel('GPS time')
    plt.ylabel('Amplitude')
    plt.savefig('data_segment.png')
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
def creat_workin_directory(sim_id, over_write):
    try :
        #print('Creating directory '+sim_id)
        os.mkdir(sim_id)
        ## creat director
    except OSError :
        ## It exists ? Remove it and create anyway
        if over_write=='y' :
            print(sim_id+' Exists')
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
def accum_results_from_condor_run(base_dir) :
	pass
    ## check for the error
##############################################################################
class network:

    """ A network of ground based gw detectors
    """
    def __init__(self, ifo_list, reference_detector = None):
        """ Create class representing a network of ground based gw detectors
        Parameters
        ----------
        ifo_list: list
            Containing some two-character detector strings, i.e. H1, L1, V1, K1, I1
        reference_detector: str
            One of the entries of ifo_list (optional)
        """
        self.ifo = ifo_list.copy()
        self.ifo_list = ifo_list.copy()
        if reference_detector == None:
            self.rifo = self.ifo[0]
        else:
            self.rifo = reference_detector

    def generate_detectors(self):                           # This will be modified in future versions.
        '''
        Checks if any inconsistency is observed in realizing ifo_list feasibility.
        '''
        if self.rifo not in self.ifo:
            raise Exception('reference_detector not in ifo_list!')
        else:
            self.ridx = self.ifo.index(self.rifo)
            if all(x in self.ifo for x in ['H1','H2']) is True:
                raise Exception('Here, the network should have detectors at different sites..')
            else:
                for ii in range(len(self.ifo)):
                    if self.ifo[ii] not in [x[0] for x in get_available_detectors()]:
                        raise Exception('That is not a valid detector:',self.ifo[ii])
                    else:
                        self.ifo[ii] = Detector(self.ifo[ii])
                self.rifo = self.ifo.pop(self.ridx)
                self.Noof_det = len(self.ifo)

    def compute_time_of_flight(self):
        '''
        This function computes the travel time for light between the reference_detector
        and any other detector and stores them in a diagonal matrix.
        '''
        tof=np.zeros(self.Noof_det)
        for ii in range(self.Noof_det):
            tof[ii]=self.rifo.light_travel_time_to_detector(self.ifo[ii])
        self.tof=np.diag(tof)

    def compute_network_axes(self):
        '''
        This function computes the network coordinate axes (defined in the following
        way) w.r.t the earth fixed coordinates and the change of basis matrix:

        (1) ref_ifo (Dref) is at origin;
        (2) Dref--->D1 is z-axis;
        (3) Dref--->D1 and Dref--->D2 form xz-plane with x-component of Dref--->D2 always positive.
        '''
        zax = self.ifo[0].location - self.rifo.location
        self.zax = zax/np.linalg.norm(zax)
        if self.Noof_det >= 2:
            xax = self.ifo[1].location - self.rifo.location
            sax = xax/np.linalg.norm(xax)
            xax = xax - xax.dot(self.zax) * self.zax
            self.xax = xax/np.linalg.norm(xax)
            self.yax = np.cross(self.zax,self.xax)
            self.change_basis = np.array((self.xax, self.yax, self.zax)).transpose()

    def compute_alpha(self, idx1, idx2):
        '''
        Calculates angle bw Dref--->Di and Dref--->Dj.
        '''
        disp1 = self.ifo[idx1].location-self.rifo.location
        disp2 = self.ifo[idx2].location-self.rifo.location
        unit1 = disp1/np.linalg.norm(disp1)
        unit2 = disp2/np.linalg.norm(disp2)
        cosalpha = np.dot(unit1, unit2)
        return np.arccos(np.clip(cosalpha,-1,1))

    def compute_psi(self, idx):
        '''
        Angle bw x-axis and projection of Di onto xy-plane.
        '''
        if idx == 0 or idx == 1:
            return 0.0
        if idx >= 2:
            a = self.compute_alpha(1,idx)
            b = self.compute_alpha(0,1)
            c = self.compute_alpha(0,idx)
            cospsi = (np.cos(a) - np.cos(b) * np.cos(c))/(np.sin(b) * np.sin(c))

            disp = self.ifo[idx].location - self.rifo.location
            proj = disp - self.zax.dot(disp) * self.zax
            if np.dot(proj, self.yax) < 0:                  # To ensure psi is measured from x-axis to proj onto xy-plane
                return -np.arccos(np.clip(cospsi,-1,1))
            else:
                return np.arccos(np.clip(cospsi,-1,1))

    def compute_network_orientation(self):
        '''
        Computes a matrix that encodes the geometric orientation of all the detectors present.
        '''
        self.det_ori = np.zeros((self.Noof_det,3))
        if self.Noof_det >= 2:
            for ii in range(self.Noof_det):
                alpha = self.compute_alpha(0,ii)
                psi = self.compute_psi(ii)
                self.det_ori[ii,0] = np.sin(alpha) * np.cos(psi)
                self.det_ori[ii,1] = np.sin(alpha) * np.sin(psi)
                self.det_ori[ii,2] = np.cos(alpha)
        self.det_ori=np.nan_to_num(self.det_ori, nan=0)

    def create_time_delay_array(self, ra, dec, t_gps):
        '''
        Calculates and returns an array containing time delays for a GW source
        at ra and dec angles that passes at t_gps.
        '''
        time_delay = np.zeros(self.Noof_det)
        for ii in range(self.Noof_det):
            time_delay[ii] = self.rifo.time_delay_from_detector(self.ifo[ii], ra, dec, t_gps)
        return time_delay

    def arrival_times_to_time_delay(self, arr_times):
        '''
        Converts times of arrival to time delays w.r.t reference_detector.
        '''
        if len(arr_times) == (self.Noof_det + 1):
            ref_toa = arr_times[self.ridx]
            time_delay = ref_toa - np.delete(arr_times,self.ridx)
            return time_delay
        else:
            raise Exception('Len of arr_times should be same as ifo_list')

    def compute_sky_location(self, travel_times):
        '''
        Computing theta and phi in network frame.
        '''
        a = np.linalg.lstsq(self.tof, travel_times, rcond=None)[0]
        if self.Noof_det >= 3:
            self.source_loc, residue, rank, singular = np.linalg.lstsq(self.det_ori, a, rcond=None)
        if self.Noof_det == 2:
            self.theta = np.arccos(np.clip(a[0],-1,1))
            self.theta = np.pi/2 - self.theta
            alpha = self.compute_alpha(0,1)
            num = a[1] - np.cos(alpha) * a[0]
            din = np.sin(alpha) * np.sqrt(1 - a[0]**2)
            self.phi1 = np.arccos(np.clip(num/din,-1,1))                   # Here, cannot assign any sign to degenerate phi
            self.phi2 = 2 * np.pi - np.arccos(np.clip(num/din,-1,1))
            rx, ry, rz = spherical_to_cartesian(1, self.theta, self.phi1)
            self.source_loc1 = np.array([rx, ry, rz])
            rx, ry, rz = spherical_to_cartesian(1, self.theta, self.phi2)
            self.source_loc2 = np.array([rx, ry, rz])
        if self.Noof_det == 1:
            self.theta = np.arccos(np.clip(a[0]))
            self.phi = nan

    def transform_to_earth_coordinates(self, t_gps):
        '''
        Transforming to earth fixed coordinates.
        '''
        xax = np.array([1,0,0])
        yax = np.array([0,1,0])
        zax = np.array([0,0,1])
        if self.Noof_det >= 3:
            self.source_loc = self.change_basis.dot(self.source_loc)
            proj = self.source_loc - zax.dot(self.source_loc) * zax
            '''
            ra = np.arctan2(self.source_loc[1], self.source_loc[0])
            dec = np.pi/2 - np.arccos(np.clip(self.source_loc[2],-1,1))
            '''
            rad_distance, dec, ra = cartesian_to_spherical(self.source_loc[0], self.source_loc[1], self.source_loc[2])
            dec=dec.value
            ra=ra.value

            if np.dot(yax, proj) > 0:
                ra = ra + self.rifo.gmst_estimate(t_gps)
            else:
                ra = np.pi * 2 + ra + self.rifo.gmst_estimate(t_gps)
            ra = ra % (np.pi * 2)
            return ra, dec
        if self.Noof_det == 2:
            self.source_loc1 = self.change_basis.dot(self.source_loc1)
            self.source_loc2 = self.change_basis.dot(self.source_loc2)
            proj1 = self.source_loc1 - zax.dot(self.source_loc1) * zax
            proj2 = self.source_loc2 - zax.dot(self.source_loc2) * zax
            ra1 = np.arctan2(self.source_loc1[1], self.source_loc1[0])
            ra2 = np.arctan2(self.source_loc2[1], self.source_loc2[0])
            dec1 = np.pi/2 - np.arccos(np.clip(self.source_loc1[2],-1,1))
            dec2 = np.pi/2 - np.arccos(np.clip(self.source_loc2[2],-1,1))
            if np.dot(yax, proj1) > 0:
                ra1 = ra1 + self.rifo.gmst_estimate(t_gps)
            else:
                ra1 = np.pi * 2 + ra1 + self.rifo.gmst_estimate(t_gps)
            if np.dot(yax, proj2) > 0:
                ra2 = ra2 + self.rifo.gmst_estimate(t_gps)
            else:
                ra2 = np.pi * 2 + ra2 + self.rifo.gmst_estimate(t_gps)
            ra1 = ra1 % (np.pi * 2)
            ra2 = ra2 % (np.pi * 2)
            return np.array([ra1, ra2]), np.array([dec1, dec2])

        if self.Noof_det == 1:
            return self.phi, self.theta

    def initialize_network(self):
        '''
        No need to repeat these steps for same network config.
        '''
        self.generate_detectors()
        self.compute_time_of_flight()
        self.compute_network_axes()
        self.compute_network_orientation()

    def standard_sky_localization(self, travel_times, t_gps):
        '''
        Returns the final ra and dec angles for each injection
        with the network config.
        '''
        self.compute_sky_location(travel_times)
        ra, dec = self.transform_to_earth_coordinates(t_gps)
        return ra, dec

    def compute_analytical_phases(self, ra, dec, pol, inc, t_gps):
        '''
        Computes the binary phase as observed by the IFOs
        in the network analytically.
        '''
        Ndet=len(self.ifo_list)
        self.binary_phases=np.zeros(Ndet)
        for kk in range(Ndet):
            d=Detector(self.ifo_list[kk])
            Fp,Fc=d.antenna_pattern(ra,dec,pol,t_gps)
            num=2*Fc*np.cos(inc)
            den=Fp*(1+(np.cos(inc))**2)
            phase=np.arctan2(num,den)
            self.binary_phases[kk]=phase%(2*np.pi)
        self.rphs=self.binary_phases[self.ridx]
        self.binary_phases=np.delete(self.binary_phases,self.ridx)
        self.binary_phases-=self.rphs
        return self.binary_phases

    def arrival_phase_to_phase_diff(self, arrival_phases):
        '''
        Takes arrival phases and returns phase differences.
        '''
        if len(arrival_phases) == (self.Noof_det + 1):
            ref_phase = arrival_phases[self.ridx]
            phase_diff = np.delete(arrival_phases, self.ridx)
            phase_diff -= ref_phase
            return phase_diff
        else:
            raise Exception('Len of arr_times should be same as ifo_list')


## End of network ##
##############################################################################
class skymap:
    '''
    Need to provide: IFOs, ToAs, SNRs, gbest ToAs, gbest SNRs
    and sim_insp for initialization.
    (Current version supports the order given in skyloc_condor.py)
    '''
    def __init__(self, inj_id, ifos, toas, snrs, sim_insp, skydir):
        self.inj_id =inj_id
        self.toas = toas.copy()
        self.snrs = snrs.copy()
        id = np.argsort(self.snrs, axis=3)
        self.snrs = np.take_along_axis(self.snrs, id, axis=3)
        self.toas = np.take_along_axis(self.toas, id, axis=3)
        self.ifos = ifos.copy()
        self.ndetectors = len(self.ifos)
        self.nswarms = len(self.toas[0,:,0,0])
        self.nsteps = len(self.toas[0,0,:,0])
        self.nparticles = len(self.toas[0,0,0,:])
        self.sim_insp = sim_insp
        self.inj_ra = self.sim_insp.longitude
        self.inj_dec = self.sim_insp.latitude
        self.trig_time = self.sim_insp.geocent_end_time
        self.skydir = skydir
        self.header = 'PSO search with N$_{swarm}$ = '\
                        +str(self.nswarms)+', N$_{particle}$ = '\
                        +str(self.nparticles)+', N$_{step}$ = '+str(self.nsteps)

    def generate_network(self, vir_det=False):
        '''
        Try to virtualize each swarm as a colocated-coaligned detector.
        (Default is false.)
        '''
        self.vir_det = vir_det
        if self.vir_det:
            self.ifos *= self.nswarms
            self.toas = self.toas.reshape(self.ndetectors*self.nswarms,\
                                            self.nsteps,self.nparticles,order='F')
            self.snrs = self.snrs.reshape(self.ndetectors*self.nswarms,\
                                            self.nsteps,self.nparticles,order='F')
        else:
            self.toas = np.mean(self.toas, axis=1)
            self.snrs = np.mean(self.snrs, axis=1)
        self.ndetectors = len(self.ifos)

    def impose_snr_cut(self, threshold=0):
        '''
        Restricts IFOs without enough SNR to take part in sky loc.
        (Default is no restriction)
        '''
        self.threshold = threshold
        final_gbest_snrs = np.amax(self.snrs[:, self.nsteps-1,:], axis=1)
        cut_id = np.argwhere(final_gbest_snrs < threshold)
        while len(cut_id)>1:    # Fix this later
            self.threshold-=0.01
            cut_id = np.argwhere(final_gbest_snrs < self.threshold)
        self.ifos = list(np.delete(self.ifos, cut_id))
        self.toas = np.delete(self.toas, cut_id, axis=0)
        self.snrs = np.delete(self.snrs, cut_id, axis=0)
        self.net = network(self.ifos)
        self.net.initialize_network()

    def set_proper_angle_range(self, ra, dec):
        ra = (ra*180.0/(np.pi)) * u.degree
        dec = (dec*180.0/np.pi) * u.degree
        sky_c = SkyCoord(ra=ra, dec=dec, frame='icrs')
        ra = sky_c.ra.wrap_at(180 * u.deg).radian
        dec = sky_c.dec.radian
        return ra, dec

    def plot_skymap(self, step=None, fname=None):
        '''
        Plots a static skymap in Mollweide projection.
        '''
        ra_region = np.zeros(self.nparticles)
        dec_region = np.zeros(self.nparticles)
        snr_region = np.zeros(self.nparticles)
        fig=plt.figure(figsize=(12, 8))
        ax=plt.subplot(projection='mollweide')
        ra, dec = self.set_proper_angle_range(self.inj_ra, self.inj_dec)
        ax.plot(ra, dec, 'k*', label='True location',markersize=12)
        if step is None:
            step = self.nsteps - 1
        else:
            ax.text(x=4.5,y=2,s='Current step: {}'.format(step))
        local_snrs = self.snrs[:,step,:]
        local_toas = self.toas[:,step,:]
        for ii in range(self.nparticles):
            time_delay = self.net.arrival_times_to_time_delay(local_toas[:,ii])
            ra_region[ii], dec_region[ii] = self.net.standard_sky_localization(time_delay,self.trig_time)
            if self.vir_det:
                snr_region[ii] = np.sqrt(np.sum(local_snrs[:,ii]**2))/np.sqrt(self.nswarms)
            else:
                snr_region[ii] = np.sqrt(np.sum(local_snrs[:,ii]**2))
        if self.vir_det:
            coinc_snr = np.sqrt(np.sum(local_snrs[:,ii]**2))/np.sqrt(self.nswarms)
        else:
            coinc_snr = np.sqrt(np.sum(local_snrs[:,ii]**2))
        ax.text(x=3.5,y=1,s='G-best coinc SNR: {}'.format(np.round(coinc_snr,2)))
        time_delay = self.net.arrival_times_to_time_delay(local_toas[:,ii])
        self.est_ra, self.est_dec = self.net.standard_sky_localization(time_delay,self.trig_time)
        ra, dec = self.set_proper_angle_range(self.est_ra, self.est_dec)
        ax.plot(ra, dec, 'gx', label='Estimated location',markersize=12)
        ra, dec = self.set_proper_angle_range(ra_region, dec_region)
        s = plt.scatter(ra, dec, c = snr_region, cmap=plt.cm.Spectral)
        c = plt.colorbar(s,ax=ax,label='Coincident SNR',orientation='horizontal',aspect=50)
        #plt.clim(self.threshold, 25)   #Fix this later
        try:
            sns.kdeplot(x=ra,y=dec,levels=[0.1],color='k')
        except np.linalg.LinAlgError:
            print('No KDE plot possible here!')
        ax.grid()
        ax.legend(loc='upper left')
        ax.set_xlabel('Right ascension')
        ax.set_ylabel('Declination')
        ifo_names = [''.join([x[0] for x in np.unique(self.ifos)])]
        plt.title('Skymap with IFOs: '+str(ifo_names)+' \n'+self.header)
        plt.tight_layout()
        if fname is not None:
            fname = self.skydir+'/skymap_{}_{}.png'.format(self.inj_id,step)
            plt.savefig(fname, dpi=200)
        plt.close()

    def save_skymap(self, make_movie=False, store_steps=False):
        '''
        Saves static skymaps or makes a movie.
        '''
        if make_movie is False:
            step=self.nsteps-1
            fname=self.skydir+'/'+'skymap_'+str(self.inj_id)+'_'+str(step)+'.png'
            self.plot_skymap(step, fname=fname)
        else:
            import imageio
            images = []
            filenames=[]
            for ii in range(self.nsteps):
                fname=self.skydir+'/'+'skymap_'+str(self.inj_id)+'_'+str(ii)+'.png'
                self.plot_skymap(step=ii, fname=fname)
                filenames.append(fname)
            for filename in filenames:
                images.append(imageio.imread(filename))
                if store_steps is False:
                    os.remove(filename)
            imageio.mimsave(self.skydir+'/'+'skymap_{}.gif'.format(self.inj_id), images, duration=1)
## End of skymap ##
##############################################################################
if __name__ == "__main__":
    print('Module only, no main')
##############################################################################
## Project contributers
## Anuradha Samajdar (Was PhD student at IISER Kolkata)
## Varun Srivastava  (Was BS-MS student at IISER Pune, Grad student Syracuse )
## Shubhagata Bhaumik (Was BS-MS student at IISER Kolkata)
## Ankit Mandal  (Was BS-MS student at IISER Kolkata)
## Tathagata Pal (Was BS-MS student at IISER Kolkata)
## Aritra Aich   (Was BS-MS student at IISER Kolkata)
## Souradeep Pal (Is Grad student at IISER Kolkata)
## Gopi Patel (Was MR by research student at IISER Kokata)
## Shobhit Ranjan (Did IAS summer project 2021)
## Rajesh Kumble Nayak (One of the PI, IISER Kolkata)
## Sukanta Bose (One of the PI, IUCAA Pune)
