close all
clear
%clc


%filePath = 'C:\Users\Mikkel\ABE_Master_thesis\PythonFiles\thinLinc\Mag\CNN\wavPredsForMultiPESQ\';
%addpath 'C:\Users\Mikkel\ABE_Master_thesis\PythonFiles\thinLinc\Mag\CNN\wavPredsForMultiPESQ\';
%filePath = 'C:\Users\Mikkel\ABE_Master_thesis\PythonFiles\thinLinc\Mag\CNN\wavPredictionFiles\';
%addpath 'C:\Users\Mikkel\ABE_Master_thesis\PythonFiles\thinLinc\Mag\CNN\wavPredictionFiles\';



%filePath = 'C:\Users\Mikkel\ABE_Master_thesis\MatlabFiles\STOI\FFN_NoTF_WITHExp_STOI\';
%addpath C:\Users\Mikkel\ABE_Master_thesis\MatlabFiles\STOI\FFN_NoTF_WITHExp_STOI\;

%filePath = 'C:\Users\Mikkel\ABE_Master_thesis\MatlabFiles\STOI\FFN_NoTF_NoExp\';
%addpath C:\Users\Mikkel\ABE_Master_thesis\MatlabFiles\STOI\FFN_NoTF_NoExp\;

%filePath = 'C:\Users\Mikkel\ABE_Master_thesis\MatlabFiles\STOI\CNN_NoTF_WITHExp_STOI\';
%addpath C:\Users\Mikkel\ABE_Master_thesis\MatlabFiles\STOI\CNN_NoTF_WITHExp_STOI\;

filePath = 'C:\Users\Mikkel\ABE_Master_thesis\MatlabFiles\STOI\CNN_NoTF_NoExp_STOI\';
addpath C:\Users\Mikkel\ABE_Master_thesis\MatlabFiles\STOI\CNN_NoTF_NoExp_STOI\;


wavNames = dir([filePath,'\','*.wav']);

N = numel(wavNames)/3;
%N = numel(wavNames)-2;

prePESQ = zeros(N,1);
postPESQ = zeros(N,1);

preMCC = zeros(N,1);
postMCC = zeros(N,1);

preSTOI = zeros(N,1);
postSTOI = zeros(N,1);

%preMSE = zeros(N,1);
%postMSE = zeros(N,1);

%%

for k = 1:N
    [pred,fs]    = audioread([num2str(k),'pred','.wav']);
    %[pred,fs]    = audioread(wavNames(k).name);
    %pred = sum(pred,2);
    %pred = pred./max(abs(pred));
    [feature,fs] = audioread([num2str(k),'feat','.wav']);
    %[feature,fs] = audioread('featTestMultBCM.wav');
    %feature = sum(feature,2);
    %feature = feature./max(abs(feature));
    [label,fs]   = audioread([num2str(k),'ref', '.wav']);
    %[label,fs] = audioread('refTestMultBCM.wav');
    %label = sum(label,2);
    %label = label./max(abs(label));

    %prePESQ(k) = pesq( label, feature, fs );
    %postPESQ(k) = pesq( label, pred, fs );
    
    %[preMCC(k),~] = calcSpecMCC(label,feature,fs);
    %[postMCC(k),~] = calcSpecMCC(label,pred,fs);
    
    preSTOI(k) = stoi(label,feature,fs);
    postSTOI(k) = stoi(label,pred,fs);

    %preMSE(k) = mean(sqrt((label-feature).^2));
    %postMSE(k) = mean(sqrt((label-pred).^2));    
end
%%
preMOS = mean(prePESQ);
postMOS = mean(postPESQ);

plotMOS = postPESQ-prePESQ;
deltaMOS = postMOS - preMOS;

std_preScore = std(prePESQ);
std_postScore = std(postPESQ);

preMeanMCC = mean(preMCC);
postMeanMCC = mean(postMCC);

plotMCC = postMCC-preMCC;
deltaMCC = postMeanMCC - preMeanMCC;

Avag_preSTOI = mean(preSTOI);
Avag_postSTOI = mean(postSTOI);

plotSTOI = postSTOI-preSTOI;
deltaSTOI = Avag_postSTOI - Avag_preSTOI;

%deltaMSE = mean(postMSE - preMSE);
%plotMSE = postMSE - preMSE;

disp(['PESQ improvement: ',num2str(deltaMOS)])
disp(['MCC  improvement: ',num2str(deltaMCC)])
disp(['STOI  improvement: ',num2str(Avag_postSTOI)])
%disp(['MSE  improvement: ',num2str(deltaMSE)])

%%

figure(1)
subplot 311
plot(plotMOS,'*','linewidth',2)
ylabel('MOS score')
xlabel('File nr.')
title(['Delta MOS scores, mean: ',num2str(deltaMOS)])
%xlim([1 length(plotMOS)])
grid on

subplot 312
plot(plotMCC,'*','linewidth',2)
ylabel('MCC score')
xlabel('File nr.')
title(['Delta MCC scores, mean: ',num2str(deltaMCC)])
grid on

subplot 313
plot(plotSTOI,'*','linewidth',2)
ylabel('LSD score')
xlabel('File nr.')
title(['Delta STOI scores, mean: ',num2str(deltaSTOI)])
grid on

%subplot 414
%plot(plotMSE,'-o','linewidth',1)
%ylabel('MSE score')
%xlabel('File nr.')
%title(['Delta MSE scores, mean: ',num2str(deltaMSE)])
%grid on