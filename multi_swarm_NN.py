import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt 
import pickle	
import gc
import time
from datetime import datetime
from pycbc.types import TimeSeries, FrequencySeries, Array, float32, float64, complex_same_precision_as, real_same_precision_as
## OS related imports
import sys
import os
import shutil
import subprocess
import getopt
import importlib
## pycbc imports
from pycbc.waveform import get_fd_waveform
from pycbc.filter import match
from pycbc.psd import aLIGOZeroDetHighPower,  AdvVirgo, KAGRA
from pycbc.noise.gaussian import frequency_noise_from_psd
from pycbc import frame
## PSO import 
## our main engin!
import swarmpi_NN as swampi
import csv

if __name__ == "__main__":
	''' 
		Main function: 
		
	'''

	sim_id='multi_pso_out_0058_SN'
	## name of the output directory

	inputfile='pso_run_input_parameters_NN'
	## name of the input PSO run parameters
	try:
		opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print('multi_swarm.py  -i <inputfile> -o <outputfile>')
		sys.exit(-1)
	for opt, arg in opts:
		if opt == '-h':
			print('multi_swarm.py  -i <inputfile> -o <outputfile>')
			sys.exit(0)
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			sim_id = arg
	## Command line for input and output
	print('Input file is:', inputfile)
	print('Output file is:', sim_id)
	## finaly print the option on the screen. the Log file is not ready yet
	########################################################################
	input_mod=importlib.import_module(inputfile)
	globals().update(input_mod.__dict__)
	#from pso_run_input_parameters import *
	# input parameter file is loaded as module!! 
	## Is there a better way ?
	#######################################################################
	print(datetime.now().ctime(), ': Starting the simulation')
	tlen=int(tmax*signal_sps) 
	flen=tlen/2+1
	delta_f=1.0/tmax
	sim_id=swampi.creat_workin_directory(sim_id, over_write)
	print (sim_id)
	if sim_id=='' :
		print('Could not open directory')
		sys.exit(-1)
	#print 'Chaning to :', sim_id
	log_file_name=sim_id+'.log'
	os.chdir(sim_id)
	## Now it is time to create log file!
	log=swampi.pso_log(log_file_name)
	log.write('Chaning to :'+sim_id)
	inj_list=[]
	log.write('#'*20)
	log.write('Generating Injection')
	for ii in range(Ninj) :
		if ii == 0 :
			inj_list.append(swampi.inj_par('IMRPhenomPv2', inj_seed=inj_seed, ifo=ifo, 
					search_dim=search_dim, noise_seed=noise_seed))
		else :
			inj_list.append(swampi.inj_par('IMRPhenomPv2', inj_seed=None, ifo=ifo, 
					search_dim=search_dim, noise_seed=noise_seed))
		inj_list[ii].signal_stat=sign_stat
		inj_list[ii].st_time=st_time
		inj_list[ii].trig_time=trig_time[ii]
		inj_list[ii].frame_file_name=frame_file_name 
		inj_list[ii].frame_ch=frame_ch
		inj_list[ii].O2_PSD_file=O2_PSD_file
		
	m1_expected = np.zeros(Ninj)
	m2_expected = np.zeros(Ninj)
	m1_obtained = np.zeros(Ninj)
	m2_obtained = np.zeros(Ninj)
	snr = np.zeros(Ninj)
	toa = np.zeros(Ninj)
	distance = np.zeros(Ninj)
	#################################################### Variables for NN
	chirp_mass=np.zeros((nparticales,nsteps))
	pmass1=np.zeros((nparticales,nsteps))
	pmass2=np.zeros((nparticales,nsteps))
	swm1=np.zeros((nparticales,nsteps,Noof_swarms)) ### for multiple swarms
	swm2=np.zeros((nparticales,nsteps,Noof_swarms))
	pmatch=np.zeros((nparticales,nsteps))
	############################################################################
	for num_of_inj in range(Ninj) :
	
		log.write('#'*20)
		log.write('Injection '+str(num_of_inj))
		## Creating sub directory for each injection
		s1 = sim_id+'_'+str(num_of_inj).rjust(4,'0')
		## We  have to creat at directory for each injection  if writing is necessary
		try :
			os.mkdir(s1)
			## creat directory
		except OSError :
			## It exists ? Remove it and create anyway
			shutil.rmtree(s1)
			os.mkdir(s1)
		log.write("changing to: "+s1)
		print("changing to: "+s1)
		os.chdir(s1)
		try :
			log.write('Creating plots directory')
			os.mkdir(plots_dir)
			## creat directory
		except OSError :
			## It exists ? Remove it and create anyway
			log.write('Over writing plot directory')
			shutil.rmtree(plots_dir)
			os.mkdir(plots_dir)
		
		try :
			log.write('Creating Data point directory')
			os.mkdir(data_points_dir)
			## creat directory
		except OSError :
			## It exists ? Remove it and create anyway
			log.write('Over writing Data point directory')
			shutil.rmtree(data_points_dir)
			os.mkdir(data_points_dir)
		#########################################################################
		row=inj_list[num_of_inj]                ## read the injectionfile from xml file
		log.write('Generating signal Waveform for the injection '+str(num_of_inj))
		#segment, psd=swampi.generate_freqdomain_strain(row) 
		segment, psd=swampi.generate_timedomain_strain(row) 
		#print('phase_from_frequencyseries= ', p)
		#seg,pd,phase=swampi.generate_timedomain_strain(row)
		#print('phase_from_polarization= ',phase)
		## gerenate strain for given ifo in frequency domain
		file_fd = open(sim_id+'_workfile_'+str(num_of_inj).rjust(4,'0')+'.out', 'w')
		print ( 'injected parameter   m1=', row.mass1,file=file_fd, end='') 
		print ( '\t m2=', row.mass2, file=file_fd, end='')
		print ( '\t D=',  row.distance, file=file_fd, end='')
		print ('injected parameter   m1=', row.mass1, end='') 
		print ('\t m2=', row.mass2, end='')
		print ('\t D=',  row.distance)
		if save_input_time_seg=='y' or save_input_time_seg=='Y' :
			#plt.loglog(np.abs(segment.sample_frequencies), np.abs(segment),'b',label='Noise+Signal') #
			plt.loglog(np.abs(psd.sample_frequencies), np.sqrt(np.abs(psd)),'r',label='S_h(f)^{1/2}')#
			plt.xlabel('Frequency')
			plt.ylabel('H(f)')
			plt.grid()
			plt.title('Signal segment in frequency domain')
			plt.xlim(row.f_lower, 0.9*signal_sps/2)
			plt.legend()
			plt.savefig(plots_dir+'/00_signal_segment_'+str(num_of_inj).rjust(6,'0')+'.png')
			plt.close()
		#print 'panic quicting'; exit(0)
		log.write('Creating PSO swarms for Injection '+str(num_of_inj))
		swarm_list=[]
		## Generating swarmlist for multiswarm
		for kk in range(Noof_swarms) :
			swarm_list.append(swampi.swarm(sps=signal_sps,
			tmax=tmax,sim_insp_par=row, ifo=ifo, 
			flow=row.f_lower, 
			particles=nparticales, 
			steps=nsteps, 
			dim=search_dim))## initialising swarms	
			#swarm_listsw=swampi.swarm(sps=signal_sps,tmax=tmax, ifo=ifo, flow=row.f_lower) 
			if search_algo == 1 :
				swarm_list[kk].evolve=swarm_list[kk].evolve_standard_pso
				swarm_list[kk].evolve_option=evolve_option
		## This process can be serial, should be quick
		########################################################################
		## PSO evolutio for each swarm
		log.write('#'*20)
		log.write('Staring the PSO evolution for Injection '+str(num_of_inj))
		swarm_gbest_snr=np.zeros(Noof_swarms)
		swarm_gbest_pos=np.zeros((Noof_swarms,2))
		for ii in range (swarm_list[0].Nsteps):
			if verbose =='y' or verbose=='Y' :	
				print('Work in progress=',int(float(ii)/swarm_list[0].Nsteps*100),'%')
			for kk in range(Noof_swarms) :
				pmatch,swm1[:,:,kk],swm2[:,:,kk]= swarm_list[kk].evolve(segment, row)
			for jj in range(nparticales):
				pmass1[jj,ii]=sum(swm1[jj,ii,:])/Noof_swarms
				pmass2[jj,ii]=sum(swm2[jj,ii,:])/Noof_swarms
			if save_step_plot=='y' or save_step_plot=='Y' :
				swampi.plot_swarm_poistion(swarm_list,
					plots_dir+'/02_pso_plot_step'+str(swarm_list[kk].curr_step).rjust(4, '0') )
			if save_step_data =='y' or save_step_data =='Y' :
				swampi.save_swarm_poistion(swarm_list,
					data_points_dir+'/02_pso_plot_step'+str(swarm_list[kk].curr_step).rjust(4, '0') )	
			#if ii == 0 :
				#swarm_gbest_snr=swarm_list[0].gbest_snr
				#swarm_gbest_pos=np.copy(swarm_list[0].gbest)
			
			if sync_step==1 or (sync_step!=0 and ii%sync_step ==0) :
				for kk in range(Noof_swarms) :
					if swarm_list[kk].gbest_snr > swarm_gbest_snr[kk] :
						swarm_gbest_snr[kk]=swarm_list[kk].gbest_snr
						swarm_gbest_pos[kk,0]=swarm_list[kk].gbest[0]
						swarm_gbest_pos[kk,1]=swarm_list[kk].gbest[1]
		for kk in range(Noof_swarms) :
			swarm_list[kk].gbest[0]=swarm_gbest_pos[kk,0]
			swarm_list[kk].gbest[1]=swarm_gbest_pos[kk,1]
			swarm_list[kk].gbest_snr=swarm_gbest_snr[kk]
		########################################################################
		log.write('PSO evoltion Done for Injection '+str(num_of_inj))
		log.write('#'*20)
		if save_radius_plot == 'y' or save_radius_plot=='Y' :
			log.write('Plotting Swarm radius for Injection '+str(num_of_inj))
			swampi.plot_mean_distance(swarm_list,plots_dir+'/01_pso_plot_radius.png')
		if save_snr_plot == 'y' or save_snr_plot == 'Y' :
			log.write('Plotting Swarm SNR as function of step for Injection '+str(num_of_inj))
			swampi.plot_swarm_gbest_snr(swarm_list,plots_dir+'/03_pso_plot_gbest_snr.png')
		log.write('Writing results for Injection '+str(num_of_inj))
		for kk in range(Noof_swarms) :
			#print ( 'Estimated parameter     m1=', swarm_list[kk].gbest[0], '\t m2=',swarm_list[kk].gbest[1], '\t SNR=', swarm_list[kk].gbest_snr, '\t toa=', swarm_list[kk].gbest_toa, file=file_fd)
			print ( '\nEstimated parameter     m1=', swarm_list[kk].gbest[0], '\t m2=',swarm_list[kk].gbest[1], '\t SNR=', swarm_list[kk].gbest_snr, file=file_fd)
			print  ('Estimated parameter     m1=', swarm_list[kk].gbest[0], '\t m2=',swarm_list[kk].gbest[1], '\t SNR=', swarm_list[kk].gbest_snr, '\t toa=', swarm_list[kk].gbest_toa)
		
		os.chdir('..')
		
		sum_m1 = 0.0
		sum_m2 = 0.0
		sum_snr = 0.0
		sum_toa=0.0
		for kk in range(Noof_swarms):
			#sum_m1 = sum_m1 + (swarm_list[kk].gbest[0])*(swarm_list[kk].gbest_snr)
			#sum_m2 = sum_m2 + (swarm_list[kk].gbest[1])*(swarm_list[kk].gbest_snr)
			sum_m1 = sum_m1 + (swarm_list[kk].gbest[0])
			sum_m2 = sum_m2 + (swarm_list[kk].gbest[1])
			sum_snr = sum_snr + swarm_list[kk].gbest_snr
			sum_toa = sum_toa + swarm_list[kk].gbest_toa

		#m1_avg = sum_m1/sum_snr
		#m2_avg = sum_m2/sum_snr
		m1_avg = sum_m1/Noof_swarms
		m2_avg = sum_m2/Noof_swarms
		snr_avg = sum_snr/Noof_swarms
		toa_avg = sum_toa/Noof_swarms

		print('Estimated Weighted Average of m1:',m1_avg)
		print('Estimated Weighted Average of m2:',m2_avg)
		print('Estimated  Average of SNR:',snr_avg)
		print('Estimated  Average of toa:',toa_avg)

		m1_expected[num_of_inj] = row.mass1
		m2_expected[num_of_inj] = row.mass2
		m1_obtained[num_of_inj] = m1_avg
		m2_obtained[num_of_inj] = m2_avg
		snr[num_of_inj] = snr_avg
		toa[num_of_inj] = toa_avg/signal_sps
		distance[num_of_inj] = row.distance
		file_id = open('data_for_hist.txt', 'a+')
		#print(m1_expected[num_of_inj],m2_expected[num_of_inj],distance[num_of_inj],m1_obtained[num_of_inj],m2_obtained[num_of_inj],snr[num_of_inj],			trig_time[num_of_inj],toa[num_of_inj], file=file_fd)
		
		
		log.write('Cleaning up')
		for kk in swarm_list :
			del kk
		gc.collect()
		steps=nsteps
		particles=nparticales
		log.write('Creating Dataset directory')
		d1 = 'dataset'+'_'+str(num_of_inj).rjust(4,'0')
		os.mkdir(d1)
		log.write("changing to: "+'dataset')
		print("changing to: "+'dataset')
		os.chdir(d1)
		
		#dataset_file=swampi.dataset(nsteps,nparticales,chirp_mass,match)
		
	   
		with open('NN_Dataset'+str(num_of_inj)+'.csv', mode='w') as csv_file:
					fieldnames=['particle', 'step','mass1','mass2', 'match', 'Target']
					writer = csv.DictWriter(csv_file, fieldnames = fieldnames)	
					writer.writeheader()
					for ii in range(particles):
						for jj in range(steps):
				
							
							writer.writerow({'particle': ii, 'step': jj, 'mass1': pmass1[ii,jj],'mass2': pmass2[ii,jj], 'match':pmatch[ii,jj], 'Target': '1'})
		
		
		
	   
		with open('NN_test_Dataset'+str(num_of_inj)+'.csv', mode='w') as csv_file:
					fieldnames=['particle', 'step','mass1','mass2', 'match']
					writer = csv.DictWriter(csv_file, fieldnames = fieldnames)	
					writer.writeheader()
					for ii in range(particles):
						writer.writerow({'particle': ii, 'step': 1, 'mass1': pmass1[ii,1],'mass2': pmass2[ii,1], 'match':pmatch[ii,1]})
		
	'''Creating Dataset for training and testing the neural network'''
	'''
	steps=nsteps
	particles=nparticales
	log.write('Creating Dataset directory')
	d1 = 'dataset'+'_'+str(num_of_inj).rjust(4,'0')
	os.mkdir(d1)
	log.write("changing to: "+'dataset')
	print("changing to: "+'dataset')
	os.chdir(d1)
	
	#dataset_file=swampi.dataset(nsteps,nparticales,chirp_mass,match)
	
   
	with open('NN_Dataset.csv', mode='w') as csv_file:
				fieldnames=['particle', 'step','chirp_mass', 'match', 'Target']
				writer = csv.DictWriter(csv_file, fieldnames = fieldnames)	
				writer.writeheader()
				for ii in range(particles):
					for jj in range(steps):
			
						
						writer.writerow({'particle': ii, 'step': jj, 'chirp_mass': chirp_mass[ii,jj], 'match':match[ii,jj], 'Target': '0'})
	
	
	
   
	with open('NN_test_Dataset.csv', mode='w') as csv_file:
				fieldnames=['particle', 'step','chirp_mass', 'match']
				writer = csv.DictWriter(csv_file, fieldnames = fieldnames)	
				writer.writeheader()
				for ii in range(particles):
					writer.writerow({'particle': ii, 'step': 1, 'chirp_mass': chirp_mass[ii,1], 'match':match[ii,0]})
	
	os.chdir('..')
	log.write(datetime.now().ctime()+': Ending the simulation')
	log.write('#'*20)
	print(datetime.now().ctime(), ': Ending the simulation')
	'''
	
	os.chdir('..')
	log.write(datetime.now().ctime()+': Ending the simulation')
	log.write('#'*20)
	print(datetime.now().ctime(), ': Ending the simulation')
	
	##############################################################################################
	'''Creating Dataset for training the neural network'''
	'''steps=nsteps
	particles=nparticales
	try :
		log.write('Creating Dataset directory')
		os.mkdir(dataset)
	except OSError :
		## It exists ? Remove it and create anyway
		shutil.rmtree(dataset)
		os.mkdir(dataset)
	log.write("changing to: "+dataset)
	print("changing to: "+dataset)
	os.chdir(dataset)
	
	#dataset_file=swampi.dataset(nsteps,nparticales,chirp_mass,match)
	
	with open('NN_Dataset.csv', mode='w') as csv_file:
		fieldnames=['particle', 'step','chirp_mass', 'match', 'Target']
		writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
      
		writer.writerheader()
		for ii in range(particles):
			for jj in range(steps):
				writer.writerow({'particle': ii, 'step': jj, 'chirp_mass': chrip_mass[ii,jj], 'match':match[ii,jj], 'Target': '1'})'''
