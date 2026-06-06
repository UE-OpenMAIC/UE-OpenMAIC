
function [averaged_crosscount] = ...
    SegmentMultivariateTimeSeries(ts, slWindow )

    [nrows, nchannels] = size(ts);

    averaged_crosscount = zeros(1, nrows-slWindow);

    for channelInd = 1:nchannels
        disp('channel')
        channelData = ts(:,channelInd);
        [crosscount] = RunSegmentation(channelData, slWindow);
        averaged_crosscount = averaged_crosscount+crosscount;
    end

    [averaged_crosscount] = averaged_crosscount/nchannels;
