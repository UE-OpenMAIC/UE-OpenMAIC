

stepSize = 3; 


run_on('PAMAP2', 3, 16);

function [] = run_on(dataset, stepSize, subSequnceLength)
    mkdir(['output_FLOSS\crosscount\',dataset]);
    path = ['data\FLOSS_format\',dataset,'\*.txt'];
    cps_path = ['data\FLOSS_format\',dataset,'\*.cp'];
    fileID = fopen(['output_FLOSS\', dataset, '_segpos.txt'],'w');
    [vals, fnames, numfids] = readFiles(path);
    [vals_cp, fnames_cp, numfids_cp] = read_CP_Files(cps_path);
    
    
    for filesInd = 1:numfids
        t = 0;
        tic;
        groundTruthSegPos = vals_cp{filesInd}';
        disp('Working on....');
        disp(fnames(filesInd,:));
        data = vals{filesInd};
        [averaged_crosscount] = SegmentMultivariateTimeSeries(data, subSequnceLength);
        [~, n] = size(groundTruthSegPos);
        [localMinimums, indLM] = findLocalMinimums(averaged_crosscount, stepSize*subSequnceLength, n);
        [~, dataLength] = size(averaged_crosscount);
        score = calcScore(groundTruthSegPos, indLM, dataLength);
        write2File(fileID,fnames(filesInd,:),indLM, score);
        writematrix(averaged_crosscount', ['output_FLOSS\crosscount\',dataset,'\',fnames(filesInd,:)]);
        t = t + toc;
        t
    end
    fclose(fileID);
end

function  [vals, files, numfids] = read_CP_Files(path)
path = strrep(path, '*.cp', '');
files1 = dir(strcat(path,'*.cp'));
files = strvcat( files1.name );
[numfids, ~] = size(files);

vals = cell(1,numfids);
for filesInd = 1:numfids
    vals{filesInd} = importdata(strcat(path,files(filesInd,:)));
end
end

function [minV, ind]= findLocalMinimums(data, length, n)
minV(1:n) = inf;
ind(1:n) = -1;
for i=1:n
    [minV(i), ind(i)] = min(data);
    data(ind(i)-length:ind(i)+length) = inf;
end
end

function score = calcScore(groundTruth, detectedSegLoc, dataLength)
[~, n] = size(groundTruth);
[~, k] = size(detectedSegLoc);
ind(1:n) = -1;
minV(1:n) = inf;

for j = 1:1:n
    for i = 1:1:k
        if(abs(detectedSegLoc(i)-groundTruth(j)) < abs(minV(j)))
            minV(j) = abs(detectedSegLoc(i) - groundTruth(j));
            ind(j) = i;
        end
    end
end

sumOfDiff = sum(minV);
score = sumOfDiff/dataLength;
end

function  [vals, files, numfids] = readFiles(path)
path = strrep(path, '*.txt', '');
files1 = dir(strcat(path,'*.txt'));
files = strvcat( files1.name );
[numfids, ~] = size(files);

vals = cell(1,numfids);
for filesInd = 1:numfids
    vals{filesInd} = importdata(strcat(path,files(filesInd,:)));
end
end

function [groundTruthSegPos, length] = getSegmentPos(names)
segmentPos = strfind(names,'_');
[~, n] = size(segmentPos);
data = [ 0 0];
for i = 1:1:n
    if(i+1 <= n)
        data(i) = str2num(names(segmentPos(i)+1:(segmentPos(i+1)-1)));
    else
        endPos = strfind(names,'.txt');
        data(i) = str2num(names(segmentPos(i) + 1:(endPos-1))); 
    end
end
length = data(1);
groundTruthSegPos =  data(2:end);
end

function write2File(fileID, name, predictedSegment, score)
fprintf(fileID,name);
fprintf(fileID,' , ');
[~, n]= size(predictedSegment);
for i=1:1:n
    fprintf(fileID,num2str(predictedSegment(i)));
    if(i~=n)
        fprintf(fileID,'_');
    else
        fprintf(fileID,',');
    end
end
fprintf(fileID,num2str(score));
fprintf(fileID,'\n');
end
