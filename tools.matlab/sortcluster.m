function [clusterim, ind] = sortcluster(clusterim, sortvar, criterion)

nclust = numel(unique(clusterim));
val = zeros(nclust,1);
for i = 1 : nclust    
    val(i) = nanmean(sortvar(clusterim == i));    
end
msg = ['Occurrence of sortvar before reordering: ' num2str(val')];
disp(msg);    

[val, ind] = sort(val, criterion);     
% msg = ['Order of clusters with high HistGrade2G: ' int2str(ind')];
% disp(msg);

c = clusterim;
for i = 1 : nclust
    clusterim(c == ind(i)) = i;    
end

% Check reordering
for i = 1 : nclust    
    val(i) = nanmean(sortvar(clusterim == i));        
end
msg = ['Occurrence of sortvar after reordering: ' num2str(val')];
disp(msg);                
