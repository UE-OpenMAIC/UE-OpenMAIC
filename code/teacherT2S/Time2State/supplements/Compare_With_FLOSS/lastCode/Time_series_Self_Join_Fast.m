
function [MatrixProfile, MPindex] = Time_series_Self_Join_Fast(A, SubsequenceLength)
exclusionZone = round(SubsequenceLength/4);

if SubsequenceLength > length(A)/2
    error('Error: Time series is too short relative to desired subsequence length');
end
if SubsequenceLength < 4
    error('Error: Subsequence length must be at least 4');
end
if length(A) == size(A, 2)
   A = A'; 
end

MatrixProfileLength = length(A) - SubsequenceLength + 1;
MatrixProfile = zeros(MatrixProfileLength, 1);
MPindex = zeros(MatrixProfileLength, 1);
[X, n, sumx2, sumx, meanx, sigmax2, sigmax] = ...
    fastfindNNPre(A, SubsequenceLength);

pickedIdx = randperm(MatrixProfileLength);
for i = 1:MatrixProfileLength
    idx = pickedIdx(i);
    subsequence = A(idx:idx+SubsequenceLength-1);
    distanceProfile = fastfindNN(X, subsequence, n, SubsequenceLength, ...
        sumx2, sumx, meanx, sigmax2, sigmax);
    distanceProfile = abs(distanceProfile);
    
    exclusionZoneStart = max(1, idx-exclusionZone);
    exclusionZoneEnd = min(MatrixProfileLength, idx+exclusionZone);
    distanceProfile(exclusionZoneStart:exclusionZoneEnd) = inf;
    
    if i == 1
        MatrixProfile = distanceProfile;
        MPindex(:) = idx;
        [MatrixProfile(idx), MPindex(idx)] = min(distanceProfile);
    else
        updatePos = distanceProfile < MatrixProfile;
        MPindex(updatePos) = idx;
        MatrixProfile(updatePos) = distanceProfile(updatePos);
        [MatrixProfile(idx), MPindex(idx)] = min(distanceProfile);
    end
end

function [X, n, sumx2, sumx, meanx, sigmax2, sigmax] = fastfindNNPre(x, m)
n = length(x);
x(n+1:2*n) = 0;
X = fft(x);
cum_sumx = cumsum(x);
cum_sumx2 =  cumsum(x.^2);
sumx2 = cum_sumx2(m:n)-[0;cum_sumx2(1:n-m)];
sumx = cum_sumx(m:n)-[0;cum_sumx(1:n-m)];
meanx = sumx./m;
sigmax2 = (sumx2./m)-(meanx.^2);
sigmax = sqrt(sigmax2);

function dist = fastfindNN(X, y, n, m, sumx2, sumx, meanx, sigmax2, sigmax)
y = (y-mean(y))./std(y,1);                      %Normalize the query
y = y(end:-1:1);                                %Reverse the query
y(m+1:2*n) = 0;

Y = fft(y);
Z = X.*Y;
z = ifft(Z);

sumy = sum(y);
sumy2 = sum(y.^2);

dist = (sumx2 - 2*sumx.*meanx + m*(meanx.^2))./sigmax2 - 2*(z(m:n) - sumy.*meanx)./sigmax + sumy2;
dist = sqrt(dist);
