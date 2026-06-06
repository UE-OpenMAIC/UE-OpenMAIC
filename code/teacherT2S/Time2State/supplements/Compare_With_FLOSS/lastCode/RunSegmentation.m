
function [crosscount] = ...
        RunSegmentation(ts, slWindow )    
    [MatrixProfile, MPindex] = Time_series_Self_Join_Fast(ts, slWindow);
    [crosscount] = SegmentTimeSeries(slWindow, MPindex);
    [crosscount] = Norm_crosscount_all(crosscount, slWindow);
