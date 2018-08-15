function [ txy, f ] = estimateTF_H1( yREF, yBCM, fs, N )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

addpath('./detectVoiceActivity/')

yREF_norm = yREF./max(abs(yREF));
yBCM_norm = yBCM./max(abs(yBCM));

%% Plotting
% t = 0:1/fs:length(yREF_norm)/fs-1/fs;
% figure(1)
% plot(t,(yREF),'b',t,yBCM,'r','linewidth',1)
% legend('Ref','BCM')
% xlabel('Time [s]')
% %xlim([0 5])
%% End Plotting

% %% Plotting specrums
% NFFT = length(yREF_norm);
% NFFT = 2^15
% 
% YREF = abs(fft(yREF_norm,NFFT));
% YREF = 2*YREF./size(YREF,1);
% YREF = YREF(1:size(YREF,1)/2,:);
% YREF_dB = 20*log10(YREF/(2e-5));
% 
% 
% YBCM = abs(fft(yBCM_norm,NFFT));
% YBCM = 2*YBCM./size(YBCM,1);
% YBCM = YBCM(1:size(YBCM,1)/2,:);
% YBCM_dB = 20*log10(YBCM/(2e-5));
% 
% f = linspace(0,fs/2,size(YBCM,1));
% 
% 
% figure(2)
% semilogx(f,YREF_dB,f,YBCM_dB,'linewidth',1)
% xlim([100 10e3])
% title('Frequency response of BCM')
% xlabel('Frequency [Hz]')
% ylabel(['Amplitude [dB SPL]'])
% legend('Ref','BCM')
% grid on

% fc = 6000;
% 
% [b,a] = butter(10,fc/(fs/2),'Low');

%yREF_norm = filter(b,a,yREF_norm);
%yBCM_norm = filter(b,a,yBCM_norm);

%[ yREF_norm, yBCM_norm ] = SpeechExtraction_RefMic( yREF_norm, yBCM_norm, fs );

% w = hamming(2048);
% 
% R = round(0.25*length(w));
% M = pow2(nextpow2(length(w)));
% 
% 
% [X,dT,dF] = stft(yREF_norm,fs,w,R,M);
% [Y,dT,dF] = stft(yBCM_norm,fs,w,R,M);
% t = 0:dT:length(yREF_norm)/fs-1/fs;
% f3 = 0:dF*2:fs/2-dT;
% %%
% figure(11)
% subplot 121
% imagesc(t,f3,mag2db(abs(X(1:1024,:))))
% axis xy
% ylabel('Frequency [Hz]')
% xlabel('Time [s]')
% title('Ref mic')
% colorbar
% xlim([0 5])
% ylim([0 8e3])
% caxis([-60 0])
% 
% 
% subplot 122
% imagesc(t,f3,mag2db(abs(Y(1:1024,:))))
% axis xy
% ylabel('Frequency [Hz]')
% xlabel('Time [s]')
% title('BCM')
% colorbar
% xlim([0 5])
% ylim([0 8e3])
% caxis([-60 0])

[txy,f] = tfestimate(yREF_norm,yBCM_norm,N,N/2,N,fs,'twosided');
end

