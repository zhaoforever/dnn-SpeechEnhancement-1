function [ x] = adjustSNR( signal, noise, snrdB )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

C = sqrt(var(signal)/(var(noise)*10^(snrdB/10)));

x = signal + C*noise;

end

