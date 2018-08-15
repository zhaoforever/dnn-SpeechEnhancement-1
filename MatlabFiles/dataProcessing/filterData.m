close all
clear
clc
%%
%fl = 300;
%fh = 2500;
%[b,a] = butter(8,[fl fh]/(fs/2));

%fc = 3000;
%fs = 16000;

%[b,a] = butter(4,[fc]/(fs/2));


%filePathBCM = 'C:\Users\s123028\realRecs\features1\';
%filePathREF = 'C:\Users\s123028\realRecs\labels1\';
%filePath = 'C:\Users\s123028\dataset1\TIMIT_train_ref';

load('meanBcmTXY')

b = fftshift(ifft(meanFullTXY));
a = 1;
% 
% figure(1)
% freqz(b,a)
% 
% [ref,fs] = audioread([filePath,'\952_ref.wav']);

% 
% ref = ref(1:fs*5);
% y = y(1:fs*5);
% x = filter(fftshift(ifft(meanFullTXY)),1,ref);
% 
% y = y./max(abs(y));
% x = x./max(abs(x));
% 
% [Y,dt,df] = stft(y,fs);
% [X,dt,df] = stft(x,fs);
% 
% f = 0:df:fs-df;
% t = linspace(0,length(x)/fs-dt,size(Y,2));
% 
% figure(1)
% surf(t,f,log(abs(real(Y))),'EdgeColor','none')
% ylim([0 8e3])
% title('Real')
% axis xy; view(0,90);
% 
% min(min(imag(Y)))
% 
% figure(2)
% surf(t,f,log10(imag(Y)),'EdgeColor','none')
% ylim([0 8e3])
% title('Imag')
% axis xy; view(0,90);
% 
% 
% preScore = pesq( ref, x, fs )
%%
filePathFeat = 'C:\Users\s123028\dataset10\feat\';
filePathLabel = 'C:\Users\s123028\dataset10\label\';
dN = 1023;
wavNames = dir([filePathFeat,'*.wav']);

for i = 1:numel(wavNames)
   [x,fs] = audioread([filePathFeat,wavNames(i).name]);
   [y,fs] = audioread([filePathlabel,wavNames(i).name]);
   
   x = resample(x,1,3);
   y = resample(y,1,3);
   
   x = filter(b,a,x);
   dN = finddelay(x,y);
   x = [x;zeros(dN,1)];
   y = [zeros(dN,1);y];
    
   %newFileName = strrep(wavNames(i).name,'ref','feat');
   audiowrite(['.\feat\',wavNames(i).name],x,fs);
   audiowrite(['.\label\',wavNames(i).name],y,fs);
   
end

%%
% [D,dT,dF] = stft(x,fs);
% [Y,dT,dF] = stft(y,fs);
% 
% f = 0:dF:fs-dF;
% t = linspace(0,length(x)/fs-dT,size(Y,2));
% 
% M = D./Y;
% %%
% figure(1)
% surf(t,f,(abs(M)),'EdgeColor','none')
% ylim([0 8e3])
% %zlim([-10 0])
% title('Label')
% axis xy; view(0,90);
% colorbar
% 
% %%
% close all
% %x = x./max(abs(x));
% %y = y./max(abs(y));

figure(1)
plot(x)
hold on
plot(y)
%xlim([2500 10000])