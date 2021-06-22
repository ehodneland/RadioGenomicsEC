% Path to folder where the patient data are
printresultpath = 'results';
printfigspath = 'figs';

% Path to cluster file
pathcluster = 'results/analysegeneticdata-METHOD-all-NCLUSTERS-3.mat';

% Find clusters
% clusteroption = 'createcluster';
clusteroption = 'applycluster';

% Number of clusters
nclust = 3;

% Subclustering?
subclust = true;

% Which data to use for analysis
dataoption = 'manual';
% dataoption = 'ML';

% Image data to load
pathimagedata = 'data/TexturevarsManual.csv';
pathimagedataML = 'data/TexturevarsML.csv';

% Which variables to cluster from
clustermethod = 'all';
% clustermethod = 'volume';
% clustermethod = 'allbutvolume';

% Name of file
mfilename = 'analyseradiomicdata';

% fontsize
fs = 16;
