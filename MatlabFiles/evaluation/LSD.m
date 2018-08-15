function [ LSD ] = LSD(ref,degraded,fs)
%LSD Log-Spectral Distance
%   Detailed explanation goes here

[X,~,~] = stft(ref,fs);
[Y,~,~] = stft(degraded,fs);

X_log = log(abs(X + 1e-7).^2);
Y_log = log(abs(Y + 1e-7).^2);

LSD = mean(mean(Y_log - X_log,1));

end

