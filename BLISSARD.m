function [ X_c, ADD] = BLISSARD( X, AC, param, flags )
% BLISSA²RD written by Alexander von Lühmann, PhD, 2018
% avonluh@gmail.com
%
% Method is based on the framework described in 
% von Lühmann, A., Boukouvalas, Z., Müller, K.-R., and Adali, T. “A new blind source separation
% framework for signal analysis and artifact rejection in functional
% Near-Infrared Spectroscopy”, 2019, p72-88, NeuroImage
% https://doi.org/10.1016/j.neuroimage.2019.06.021

%
% BLISSARD uses multimodal data, here fNIRS and acceleration signals, to
% minimize components(artifacts) that coßdepend on one of the modalities, e.g., movement.
% It performs ERBM ICA on raw fNIRS intensity signals to find independent sources.
% It then performs CCA on these sources and and synchronously acquired
% acceleration signals to find projections for each modality that have
% maximal canonical correlation. The highly correlating CCA components based
% on fNIRS signals are then backprojected and used to clean the fNIRS
% sources.
%
% Method dependends on ERBM ICA [Li and Adali], which can be found here:
% http://mlsp.umbc.edu/codes/ERBM.zip
% [Li and Adali]: X.-L. Li, and T. Adali, 
% "Blind spatiotemporal separation of second and/or higher-order correlated 
% sources by entropy rate minimization," in Proc. IEEE Int. Conf. Acoust., 
% Speech, Signal Processing (ICASSP), Dallas, TX, March 2010. 
%
% INPUTS:
% X:    input nirs data with dimensions |time x channels|
% AC:   input accel data with dimensions |time x channels|
%                    feeding in orthogonalized (PCA) data is advised.
% param.p:    ERBM filter order, use 15 per default
% param.tshift:  temporal embedding parameter (lag in samples)
% param.NumOfEmb: Number of temporally embedded copies
% param.nc:   number of constraints to be used. 3 (accel axes) by default
% param.ct:   correlation threshold. removes constrained sources before
%       backprojection only, if correlation between constraint and
%       constrained source is >ct. Use 0.4 per default
% flags.nf:   normalization flag (nf=true: normalize input data, default)
% flags.pcaf:  flags for performing pca of [X AC] as preprocessing step.
%       default: [1 1]
%
% OUTPUTS:
% X_c:  Backprojected (cleaned) data
% C:    fNIRS constraints from cca
% ADD.W:    Mixing matrix from ICA
% ADD.S_art: Artifactual sources
% ADD.ccac: CCA correlation coefficients between projected fNIRS sources
%       and projected accelerometer data
% ADD.acc_emb:   Temp. embedded accelerometer signals
% ADD.S:    demixed sources
% ADD.U,V:  sources in CCA space

%% Normalization
if flags.nf==true
    % normalize constraint modality (accelerometer data)
    accel_n= zscore(AC);
    % Normalize data X
   [X_in1, mean_X, std_X]= zscore(X);
else
    accel_n=AC;
    X_in1=X;
end

%% Perform PCA
if flags.pcaf(1)==true
    [coeff_afs accel_pca latent_afs] = pca(accel_n);
else
    accel_pca=accel_n;
end
if flags.pcaf(2)==true
    [coeff_x X_in latent_x] = pca(X_in1);
else
    X_in = X_in1;
end

%% Perform ERBM
W=ERBM(X_in',param.p);
ADD.W = W;
%Project to source space
S=W*X_in';

%% Plot Sources 
if flags.pf==true
    plot_components_t(S, 0, 1:size(X_in,1), [1 size(X_in,1)])
end
ADD.S=S;

%% Use CCA with sources and Accelerometer data
% and all fnirs data. result -> vector U
% create temporally embedded acceleration signal (tshift=2 samples, max 1s)
% BEWARE: Circular shifting is not optimal, use 
acc_emb=accel_pca(:,1:param.nc);
for i=1:param.NumOfEmb
    acc=circshift( accel_pca(:,1:param.nc), i*param.tshift, 1);
    acc(1:2*i,:)=repmat(acc(2*i+1,1:param.nc),2*i,1);
    acc_emb=[acc_emb acc];
    ADD.acc_emb=acc_emb;
end

%cut to same length of samples
s1=size(acc_emb,1);
s2=size(S,2);
if s1 ~=s2
    acc_emb=acc_emb(1:min([s1 s2]),:);
    S=S(:,1:min([s1 s2]));
end


%% CCA
[Wu,Wv,ccac,U,V] = canoncorr(S',acc_emb);
ADD.ccac =ccac;
ADD.U = U;
ADD.V = V;
% estimate projection matrix A   | X = A*S
Au = cov(S')*Wu/cov(U);
% set cca components to zero that have correlation < ct
compindex=find(ccac<param.ct);
Au(:,compindex)=0;
% calculate artifactual components in Sources
S_art=S'*(Au*Wu')';
ADD.S_art = S_art;
% HP filter artefactual components to prevent level-shifts
Fs = 0.5/0.06;
[d,c] = butter(3, 0.01/(Fs/2), 'high');
S_artfilt=filtfilt(d,c,S_art);
% cleaned Sources: Original - Artifacts
S_c = S - S_artfilt';

%% Backproject Sources to fNIRS space (normalized)
A = inv(W);
X_cnn= A*S_c;
X_cnn=X_cnn';

%% Add mean and std if data was normalized
if flags.nf==true
    X_c=X_cnn.*repmat(std_X',1,size(X_cnn,1))'+repmat(mean_X',1,size(X_cnn,1))';
else
    X_c= X_cnn;
end

    
end

