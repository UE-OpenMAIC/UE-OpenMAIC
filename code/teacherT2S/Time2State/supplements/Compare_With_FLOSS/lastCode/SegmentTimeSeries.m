
function [crosscount] = SegmentTimeSeries(slWindow, MPindex)
    l = length(MPindex);
    threshold = prctile(abs(MPindex - (1:l)'), 100);
    
    crosscount=zeros(1,length(MPindex)-1);
    nnmark=zeros(1,length(MPindex));
    count=0;

    for i=1:length(MPindex)
     
        if (abs(MPindex(i)-i)<=threshold)  
            small=min(i,MPindex(i));
            large=max(i,MPindex(i));
            nnmark(small)=nnmark(small)+1; 
            nnmark(large)=nnmark(large)-1;
           
        end
      
    end
    for i=1:length(MPindex)-1
        count=count+nnmark(i);
        crosscount(i)=count;
    end
