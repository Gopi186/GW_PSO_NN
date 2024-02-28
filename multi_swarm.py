#!/home/rajesh.nayak/anaconda3/envs/pycbc/bin/python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gc
import time
import pycbc.types.frequencyseries
import pycbc.types.timeseries
from datetime import datetime
from pycbc.types import TimeSeries, FrequencySeries, Array, float32, float64, complex_same_precision_as, real_same_precision_as
## OS related imports
import sys
import os
import shutil
import subprocess
import getopt
import importlib.util
## pycbc imports
from pycbc.psd import aLIGOZeroDetHighPower,  AdvVirgo, KAGRA
from pycbc import frame
import glob
## PSO import
## our main engin!
import swarmpi as swampi
print('Currently code under development for running on ldas cluster')
print('using Condor')


if __name__ == "__main__":
	'''
		Main function:

	'''
	print('command line arguments', sys.argv)
	## Let's first handle command line options here
	try:
		opts, args = getopt.getopt(sys.argv[1:],"hs:f:n:d:i:o:",["swarms=", "injxml=","injno=","ifo=","ifile=","ofile="])
	except getopt.GetoptError:
		print('multi_swarm.py  -h -s <swarm number> -f <dat_file> -n <injectionno> -d <ifo> -i <inputfile> -o <outputfile>')
		sys.exit(-1)
	for opt, arg in opts:
		if opt == '-h':
			print('multi_swarm.py  -h -s <swarms number> -f <dat_file> -n <injectionno> -d <ifo> -i <inputfile> -o <outputfile>')
			print('Currently This line')
			sys.exit(0)
		elif opt in ("-s", "--swarms") :
			swno=int(arg)
		elif opt in ("-n", "--injno"):
			injno=int(arg)
		elif opt in ("-d", "--ifo") :
			ifo=arg
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			sim_id = arg
		elif opt in ("-f", "--dfile") :
			data_file= arg
	## Command line for input and output
	print('Input file :', inputfile)
	print('Output file directory:', sim_id)
	print('Injection no :', injno)
	print('Ifo :', ifo)
	print('Swarm no:', swno)
	print('Data file :', data_file)
	## finaly print the option on the screen. the Log file is not ready yet
	########################################################################
	#input_mod=importlib.import_module(inputfile)
	spec = importlib.util.spec_from_file_location('*',inputfile )
	input_mod=importlib.util.module_from_spec(spec)
	spec.loader.exec_module(input_mod)
	globals().update(input_mod.__dict__)
	#from pso_run_input_parameters import *
	# input parameter file is loaded as module!!
	## Is there a better way ?
	#######################################################################
	print(datetime.now().ctime(), ': Starting the simulation')
	tlen=int(tmax*signal_sps)
	flen=tlen/2+1
	delta_f=1.0/tmax
	delta_t=1.0/signal_sps
	## Mess below is creating directory for storing results
	print('sim_id: output directory is', sim_id)
	## This is end of creating directory tree..
	log_file_name=sim_id+'_'+'simulationlog.txt'
	out_file_name=sim_id+'_'+'run_output_parameters.txt'
	#file_fd=open(out_file_name,'w')
	log=swampi.pso_log(log_file_name)
	log.write('Log file created '+log_file_name)
	log.write('simulation output directory is '+ sim_id)
	log.write('Command line :'+' '.join(sys.argv))
	log.write('Input file:'+inputfile)
	log.write('Output file is:'+ sim_id)
	log.write('Injection no is:'+str(injno).rjust(inj_nu_digit,'0'))
	log.write('Ifo is :'+ ifo)
	log.write('#'*20)
	################################################################################
	############################################################################
	log.write('#'*20)
	log.write('Injection '+str(injno))
	#plots_dir = sim_id+plots_dir
	#print('Directory for storing plots '+plots_dir)
	#log.write('Directory for storing plots '+plots_dir)
	data_points_dir=sim_id+'/'+data_points_dir
	print('Directory for storing data_points '+data_points_dir)
	log.write('Directory for storing data_points '+data_points_dir)
	log.write('it is assumed that these directories are created by dag creator')
	#########################################################################
	## read the injectionfile from xml file
	print('Loading signal data from the file', data_file)
	log.write('Loading signal data from the file '+data_file)
	if save_FS :
		fsegment=pycbc.types.frequencyseries.load_frequencyseries(data_file)
	elif save_TS :
		segment=pycbc.types.timeseries.load_timeseries(data_file)
		fsegment=segment.to_frequencyseries()
	print('Creating PSO swarms for Injection '+str(injno))
	log.write('Creating PSO swarms for Injection '+str(injno))
	## Generating swarmlist for multiswarm
	swarm=swampi.swarm(sps=signal_sps,
						tmax=tmax,
						ifo=ifo,
						flow=f_lower,
						particles=nparticales,
						steps=nsteps,
						dim=search_dim)## initialising swarms
	swarm.set_alpha(alpha)   ## inertial
	swarm.set_beta(beta)    ## pbest
	swarm.set_gamma(gamma)   ## gbest
	swarm.min=[m1_min,  m2_min]
	swarm.max=[m1_max, m2_max]
	swarm.run_pll=run_pll
	swarm.nof_process=nof_process



	if search_algo == 1 :
		swarm.evolve=swarm.evolve_standard_pso
		swarm.evolve_option=evolve_option
	print('swarm:',swno,' alpha=',swarm.alpha, ' beta=',swarm.beta,' gamma=',swarm.gamma)
	log.write('swarm:'+str(swno)+' alpha='+str(swarm.alpha)+' beta='+str(swarm.beta)+' 							gamma='+str(swarm.gamma))
	if save_step_data =='y' or save_step_data =='Y' :
		pos=np.zeros((swarm.Nsteps,swarm.Np,swarm.dim))
		snr=np.zeros((swarm.Nsteps,swarm.Np))
		toa=np.zeros((swarm.Nsteps,swarm.Np))
		phs=np.zeros((swarm.Nsteps,swarm.Np))
	########################################################################
        #self.p_best = np.zeros((self.Np,self.dim)) #array that stores best position for each particle
	## PSO evolutio for each swarm
	log.write('#'*20)
	print('Staring the PSO evolution for Injection '+str(injno)+ '  swarm'+str(swno))
	log.write('Staring the PSO evolution for Injection '+str(injno)+ '  swarm'+str(swno))
	if run_pll :
		print('Running parallel with',nof_process, 'processes')
		log.write('Running parallel with '+str(nof_process)+' processes')
	st=time.time()
	for ii in range (swarm.Nsteps):
		if verbose =='y' or verbose=='Y' :
			print('Work in progress=',int(float(ii)/swarm.Nsteps*100),'%')
		swarm.evolve(fsegment)
		if save_step_data =='y' or save_step_data =='Y' :
			pos[ii,:,:]=swarm.x[:,:]
			snr[ii,:]=swarm.snr[:]
			toa[ii,:]=swarm.toa[:]
			phs[ii,:]=swarm.phs[:]
	et=time.time()
	log.write('PSO:runtime'+str(et-st))
	if save_step_data =='y' or save_step_data =='Y' :
		np.save(data_points_dir+'/'+'step_data_inj_'+str(injno).rjust(inj_nu_digit,'0')+'_'+ifo_dir_base+ifo+'_'+swarm_no_base+str(swno).rjust(sw_no_digit,'0'), pos)
		np.save(data_points_dir+'/'+'snr_'+swarm_no_base+str(injno).rjust(inj_nu_digit,'0')+'_'+ifo_dir_base+ifo+'_'+swarm_no_base+str(swno).rjust(sw_no_digit,'0'), snr )
		np.save(data_points_dir+'/'+'step_gbest_inj_'+str(injno).rjust(inj_nu_digit,'0')+'_'+ifo_dir_base+ifo+'_'+swarm_no_base+str(swno).rjust(sw_no_digit,'0'),  swarm.gbest_snr_step)
		np.save(data_points_dir+'/'+'step_toa_inj_'+str(injno).rjust(inj_nu_digit,'0')+'_'+ifo_dir_base+ifo+'_'+swarm_no_base+str(swno).rjust(sw_no_digit,'0'),  swarm.gbest_toa_step)
		np.save(data_points_dir+'/'+'step_phs_inj_'+str(injno).rjust(inj_nu_digit,'0')+'_'+ifo_dir_base+ifo+'_'+swarm_no_base+str(swno).rjust(sw_no_digit,'0'),  swarm.gbest_phs_step)
		np.save(data_points_dir+'/'+'toa_'+swarm_no_base+str(injno).rjust(inj_nu_digit,'0')+'_'+ifo_dir_base+ifo+'_'+swarm_no_base+str(swno).rjust(sw_no_digit,'0'), toa)
		np.save(data_points_dir+'/'+'phs_'+swarm_no_base+str(injno).rjust(inj_nu_digit,'0')+'_'+ifo_dir_base+ifo+'_'+swarm_no_base+str(swno).rjust(sw_no_digit,'0'), phs)
	########################################################################
	log.write('PSO evoltion Done for Injection '+str(injno)+ ' swarm'+str(swno))
	log.write('#'*20)
	if save_radius_plot == 'y' or save_radius_plot=='Y' :
		log.write('Plotting Swarm radius for Injection '+str(injno))
		#swampi.plot_mean_distance(swarm_list,plots_dir+'/01_pso_plot_radius.png')
	if save_snr_plot == 'y' or save_snr_plot == 'Y' :
		log.write('Plotting Swarm SNR as function of step for Injection '+str(injno))
		#swampi.plot_swarm_gbest_snr(swarm_list,plots_dir+'/03_pso_plot_gbest_snr.png')
	log.write('Writing results for Injection '+str(injno))
	out_para=swarm.gbest.tolist()
	out_para.append(swarm.gbest_snr)
	out_para.append(swarm.gbest_toa)
	out_para=np.array(out_para)
	np.savetxt(out_file_name, out_para)
	#print ( out_para, file=file_fd)
	print  ('Estimated parameter     m1=', swarm.gbest[0], '\t m2=',swarm.gbest[1], '\t SNR=', swarm.gbest_snr, '\t toa=', swarm.gbest_toa)

	log.write(datetime.now().ctime()+': Ending the simulation')
	log.write('#'*20)
	print(datetime.now().ctime(), ': Ending the simulation')
	print('PSO runtime:',et-st)
