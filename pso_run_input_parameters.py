import numpy as np
## Some imporatant input parameters
Noof_swarms=5                # number of swarms in the search
Noof_threads=5
ifo='I1'                     # Single detector	
signal_sps=1024*4            # Sampling Rate
tmax=10.0                    # Time segment
nparticales=50
nsteps=10
search_dim=2								
verbose='y'				
sync_step=0     ## at this step they swarms will sync g_best
Ninj=1          ## Number of injections
sign_stat=2     ## 0: Gaussian noise only, 1 signal only, 2 signal+ Gaussian noise
                ## 3: O2 noise only  4: signal + O2 noise
search_algo=0   ## 0 : std PSO, 1: mass eigen
evolve_option=1 ##  only for search_algo =1 , 
                ## 0 : along both direction
                ## 1 : Along larger eigen direction
                ## 2 : Along smaller eigen direction
inj_seed=np.random.randint(1000000)  ## Or whaever your highness prefer
noise_seed=np.random.randint(1000000)
#np.random.randint(1000000)## Or whaever your highness prefer
sim_sr_no=0
## Plot saving options
save_step_plot='y'
save_input_time_seg='y'
save_radius_plot='y'
save_snr_plot='y'
save_step_data='n'
over_write='y'        ## This is output directory!	
st_time=1168519168; end_time=st_time+4096
np.random.seed(np.random.randint(100000))
trig_time=np.zeros(Ninj)
for ii in range(Ninj):
	trig_time[ii]=st_time+tmax+np.random.randint(0,4096-2*tmax)
frame_file_name='../../LIGO_Data/L-L1_GWOSC_O2_4KHZ_R1-1168519168-4096.gwf'
frame_ch='L1:GWOSC-4KHZ_R1_STRAIN'
O2_PSD_file='../../LIGO_Data/L1-AVERAGE_PSD-1163174417-604800.txt'
plots_dir='plots'
data_points_dir='data_points'
###############################################################################
#end of input parameters
