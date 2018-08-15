close all
clear
clc

addpath tools

%%

filepath = 'C:\Users\TobiasToft\TIMIT\';

subfolders = dir(filepath);
%%

w = hamming(512);
R = round(0.25*length(w));
M = pow2(nextpow2(length(w)));

comp_mat_tot = [];
for i = 3:numel(subfolders)
    subFolderName = subfolders(i).name;
    
    subSubFolders = dir([filepath, subFolderName]);
    for j = 3:numel(subSubFolders)
        %subsubFolderName = subSubFolders(j).name;
        wavNames = dir([filepath, subFolderName,'\','*.wav']);
        
        [y,fs] = audioread([filepath, subFolderName,'\',wavNames(j).name]);
                 
        x = ref2BCM( y );

        S = stft(x, fs, w, R, M);
        
        S_pos = S(1:size(S,1)/2,:);
        
        comp_mat = [real(S_pos); imag(S_pos)];
        
        
        writeNPY(comp_mat, ['.\npyTIMITFiles\',wavNames(j).name]);
    end
end








