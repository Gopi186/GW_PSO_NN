# GW_PSO_NN
GW detection and parameter estimation by employing the Particle Swarm Optimization Algorithm (PSO) and leveraging the power of Artificial Intelligence. 


Particle Swarm Optimization: 

a) Particle Swarm Optimization is a bio-inspired optimization algorithm that imitates the social behavior of animals living in colonies like herds or swarms like bird flocks, in search of food or survival as well as fish schooling. It is inspired by the social and collective behavior of these swarms and their ability to communicate with each other.

b) There is a swarm of particles i where each point denotes a position in a multidimensional space. These particles are evaluated in a specific optimization function to estimate their fitness value and store the best solution. All the particles change their position and traverse the search space according to a velocity function that considers the best position of a particle in the population (social component) and also their personal best position (nostalgia component). The particles will move in each iteration to another position till they reach an optimum position.

c) We used the PSO algorithm to run an on-the-go template generation and match computation. We used PyCBC to generate gravitational waveforms in the frequency domain and simulated signals with properties like masses that resemble CBC signals. As the PSO swarms scan the parameter space (which are the component masses of GW signal sources in our case), it keeps estimating match with the templates generated based on the mass pair corresponding to the particle’s position in the parameter space.

d) As the swarm moves, at each step, each particle tries to maximize the SNR and moves towards the global best with the velocity mentioned earlier. All the swarm particles finally converge at a position of maximum global best after traversing the parameter space, implying that it is the position corresponding to maximum SNR. Hence, the corresponding masses represent the component masses of the source of the signal simulated.

e) For Standard PSO evolution we generated 150 injections (simulated signals) in the mass range 15-40 Mo using PyCBC and performed a PSO run to detect these signals and estimate their component masses. We use the PyCBC waveform approximant IMRPhenomC to generate the signal injection and the templates. The PSO search employs five swarms with 300 particles each and iterates for 30 steps, which implies that 45K MFOs were performed to estimate one injection. We estimated the masses with a most probable error of -1.6Mo for component mass 1 and approximately 1.17Mo for component mass 2.

 
PSO Variant: Evolution along Eigen Vector:

In this method, we implement the eigenvector of covariance(correlation) matrix (ECM) method. The correlation matrix is generated using the matches corresponding to the current and next step of a PSO swarm particle. The eigenvectors generated are unequal and orthogonal to each other, the larger eigenvector represents the direction of a larger variance of data, and similarly the smaller eigenvector points towards the direction of a smaller variance. The evolution of the position of a swarm particle is along the eigenvectors of a correlation matrix.


Coincident Search:

a) There are two multi-detector detection methods: Coherent and Coincident. The coincident search method considers the detectors separately and compares individual detector statistics with their corresponding thresholds. For candidate events, the detector data is considered independently, and following that, the candidates from each of the detectors are searched for coincidences in time and possibly for other parameters. This method allows vetoing environmental or instrumental noises and efficiently recovers the sky location of the source and polarization of the signal.

b) In a coincident search, for every simulated injection the PSO evolves to detect the signal and estimate its parameter. This process takes place for each detector independently. Hence the component masses are estimated for each detector separately. The coincident SNR of the detector network is determined by calculating the quadrature summation of the individual detector SNR.

c) The average of the errors of individual detectors in estimating the component masses is around an error of 0.035 Mo for component mass1 and an error of -0.589 Mo for component mass2.


Implementing Artificial Intelligence on PSO:

a) We proposed the idea of implementing artificial intelligence on our PSO detection method, to speed up the process of detection and parameter estimation. We are trying to utilize the power of the neural network in learning the pattern of evolution of the swarm particles during a PSO search for signal detection. The neural network is trained using the positions and SNRs of each PSO swarm particle for every iteration.

b) The aim is to create a model that can classify whether a given simulated signal is a GW signal or just noise using the particle positions and particle SNRs of the first PSO iteration. Next, after successful classification, it can estimate the signal parameters using the data from the initial iterations of PSO.

c) To train the neural network to perform classification and prediction of the signal and its parameters, we need to feed the network a data set that comprises the information required. Our data set is made of 250 injections (simulated signals), out of which 200 injections are a combination of signal and Gaussian noise and 50 injections of signals comprising only Gaussian noise. For each injection, PSO is run to detect the signal and obtain the parameters. The PSO algorithm uses three swarms with 50 particles each and is run for 15 iterations, and this implies that for a single iteration, 2250 MFOs are performed, and hence for 250 injections, we have the data of 56.25K MFOs.

d) We use the MLP architecture that stands for Multi-Layer Perceptron to build our models. It is a feed-forward neural network wherein the neurons of a particular layer are connected to
 all the neurons of the next layer without any cycle. MLPs are suitable for regression prediction problems; also, they work well with tabular datasets like CSV files used in this case.

e) The Classification Mode is a binary classification model and it aims at classifying the injected signal into GW signal or noise. The output of this model is one of the target labels- 1 or 0, where 1 signifies the presence of the signal and 0 signifies absence.

f) Once the classification model classifies the injected signal as a GW signal with a probability greater than 0.9, we then move towards predicting the parameters of the detected signal. We aim to estimate the component masses and SNR of the injected signal using this model. The model is trained with 250 PSO injections that have run for 15 iterations, and the goal is to predict the parameters and SNR using only the first iteration of the PSO algorithm. In this manner, we can estimate the component masses with reduced MFOs, approximately 150 for each injection.

g) Results:  We get a maximum validation accuracy of 0.96 for our classification model. For the prediction model, the maximum validation accuracy with MeanSquaredError() as the error function is 1, and with MeanAbsoluteError() as the error function is 0.99965.

h) After plotting the confusion matrix, which is created using the test data, which is 20% of the training data set. The training data set comprises 200 (signal+noise) injections and 50 Gaussian noise injections; this implies that the noise signal data comprises 20% of the total training data and 10% of the test data. Figure 4.16 shows that 76.5% of the test data is correctly classified as GW signal, and 16.5% of the noise is correctly classified as noise, there is approximately 7% is a false negative, i.e., the signal being predicted as noise and 0%false positive, i.e., no noise is falsely detected as the signal.

i) Comparison with PSO: on comparing the error in estimating the component masses by the neural network model with the PSO algorithm for the mass range of 40-60 Mo- for the PSO algorithm the maximum number of injections lie around the error region of 8.4 Mo for component mass1 and 6.98 Mo for component mass2.

