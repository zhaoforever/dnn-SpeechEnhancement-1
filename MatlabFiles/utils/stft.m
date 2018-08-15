function [X_,dT,dF] = stft(x,fs,w,R,M)

if nargin == 2
    w = hanning(512);
    R = round(0.25*length(w));
    M = pow2(nextpow2(length(w))); 
end

% FFT length
N = length(w);

% Overlap
O = N - R;

% Number of frames
L = ceil((length(x)-O)/R);

% Time and frequency resolution
dT = L/fs;
dF = fs/N;

% Zero padding to number of frames
x = [x; zeros((O+L*R),1)];

% Indexes
idx1 = 1;
idx2 = N;

for i = 1:L
    
   x_ = x(idx1:idx2);
   
   x_ = x_.*w;
   
   x_ = [x_; zeros(M-N,1)];
   
   X_(:,i) = fft(x_)./M;
   
   idx1 = idx1+R;
   idx2 = idx2+R;
end


