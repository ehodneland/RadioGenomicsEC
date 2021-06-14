function [clusterim, centroid] = clusterselection(X, clusterim, clustval, nclust)

bw = clusterim == clustval;
X = X(bw,:);
[clusterim2, centroid] = kmeans(X,nclust,'Distance','sqEuclidean', ...
            'MaxIter', 1000, 'Replicates', 3);        
clusterim(bw) = clusterim2 + clustval - 1;   
        
