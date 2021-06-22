clear all;
close all;

% Import settings
settingsradiomicdata;

% -----------------------------------------------------------
% ------------------ Load image variables -------------------
% -----------------------------------------------------------
if isequal(dataoption, 'ML')
    msg = ['Loading image variables from ML segmentations ' pathimagedataML];
    disp(msg);
    imagevarML = readtable(pathimagedataML);
    subjML = imagevarML.Subj;
    
    msg = ['Loading image variables from manual segmentations' pathimagedata];
    disp(msg);
    imagevar = readtable(pathimagedata);
    subj = imagevar.Subj;

    % Remove common patients
    [~, ia, ~] = intersect(subjML, subj);
    imagevarML(ia,:) = [];
    
    % Reassign as image variables
    imagevar = imagevarML;
    
elseif isequal(dataoption, 'manual')
    msg = ['Loading image variables from manual segmentations' pathimagedata];
    disp(msg);
    imagevar = readtable(pathimagedata);
    
end

% Delete metastasis field, we are not going to use it
metastasis = false(size(imagevar,1),1);
try
    metastasis = imagevar.metastasis_vibe2min;
    imagevar.metastasis_vibe2min = [];
    imagevar.metastasis_ADC = [];
    imagevar.metastasis_b1000 = [];
catch
    
end

% Dont use these variables, they are from DCE-MRI
remvar = {'TTP','AUC','PE','T2'};
n = numel(imagevar.Properties.VariableNames);
remvarbw = false(n,1);
for i = 1 : numel(remvar)
   pattern = remvar{i};
   iswithin = strfind(imagevar.Properties.VariableNames,pattern);
   for j = 1 : n       
       if ~isempty(iswithin{j})
           remvarbw(j) = true;           
       end
   end    
end
fn = imagevar.Properties.VariableNames(remvarbw);
for i = 1 : numel(fn)
    imagevar.(fn{i}) = [];
end


% -------------------------------------------------------------
% Remove patients with only NaN
% -------------------------------------------------------------
n = sum(isnan(imagevar.Variables),2);
bw = n == size(imagevar.Variables, 2);
msg = ['Removed ' int2str(sum(bw)) ' patiets with only NaN in the features'];
disp(msg);
imagevar(bw,:) = [];

% Extract patient ID number (pseudo ID)
prm.subj = imagevar.Subj;
prm.nsubj = numel(prm.subj);

% Extract MR field strength
tesla = imagevar.tesla;
tesla(bw) = [];
imagevar.tesla = [];

% Remove these columns
subj = imagevar.Subj;
try
    imagevar.Subj = [];
    imagevar.Var1 = [];
    imagevar.tesla = [];
catch
   disp('Could not delete Var1 and Subj'); 
end

% Simplify the variable names
clear im
im.header = imagevar.Properties.VariableNames;
for i = 1 : numel(im.header)
    im.header{i} = strrep(im.header{i},'_','');   
    im.header{i} = strrep(im.header{i},'.','');   
end
imagevar.Properties.VariableNames = im.header;
clear im

% Change naming
imagevar.Properties.VariableNames{'volvibe2min'} = 'vol';
imagevar.Properties.VariableNames{'normsurfvolratiovibe2min'} = 'normsurfvolratio';

prm.nobs = size(imagevar.Variables,1);
prm.vars = imagevar.Properties.VariableNames;
prm.nvars = numel(prm.vars);

% Keep the initial data
imagevar0 = imagevar;

%---------------------------------------------
% Replace individual NaN valuds with its mean
%---------------------------------------------

msg = ['Running clusters with option ' clusteroption];
disp(msg);

% Remove individual NaN values with its mean
flag = false(prm.nvars,1);
for i = 1 :prm.nvars   
    fn = prm.vars{i};
    varhere = imagevar.(fn);
    meanval = nanmean(varhere);
    varhere = imagevar.(fn);
    varhere(isnan(varhere)) = meanval;
    if sum(isnan(varhere)) > 0
        flag(i) = true;
    end
    imagevar.(fn) = varhere;
end
msg = ['Number of replacements in data: ' int2str(sum(flag))];
disp(msg);

% Variables
vars = double(imagevar.Variables);

%-------------------------------------------------------------
% Load load clusters and centroids to be applied to the data
%-------------------------------------------------------------
if isequal(clusteroption, 'applycluster')    
    msg = ['Loading ' pathcluster];
    disp(msg);
    D = load(pathcluster);         
    centroid = D.centroid;
end

%---------------------------------------------
% Normalize variables for each tesla
%---------------------------------------------
if isequal(clusteroption, 'createcluster')
    [vars, means, stds] = normalizevars(vars, tesla == 1, [], []);
    T1.means = means;
    T1.stds = stds;
    [vars, means, stds] = normalizevars(vars, tesla == 2, [], []);    
    T2.means = means;
    T2.stds = stds;    
elseif isequal(clusteroption, 'applycluster')
    [vars, means, stds] = normalizevars(vars, tesla == 1, D.preprocess.T1.means, D.preprocess.T1.stds);
    T1.means = means;
    T1.stds = stds;
    [vars, means, stds] = normalizevars(vars, tesla == 2, D.preprocess.T2.means, D.preprocess.T2.stds);
    T2.means = means;
    T2.stds = stds;    
end

% Save normalization
preprocess.T1 = T1;
preprocess.T2 = T2;

% Reassign variables
imagevar.Variables = vars;

%------------------------------------------
% Cluster data based on image variables
%------------------------------------------

if isequal(clusteroption, 'createcluster')
    
    pathcluster =  fullfile(printresultpath, [mfilename '-METHOD-' clustermethod '-NCLUSTERS-' int2str(nclust) '.mat']);
    
    msg = ['Using method for clustering ' clustermethod];
    disp(msg);
    % All featuers for clustering
    if isequal(clustermethod, 'all')
        X = imagevar.Variables;
    end

    % Only tumor volume for clustering?
    if isequal(clustermethod, 'volume')    
        X = imagevar.vol;
    end

    % Take away tumor volume?
    if isequal(clustermethod, 'allbutvolume')
        X = imagevar;c
        X.vol = [];
        X = X.Variables;
    end

    % Initialize a cluster image
    clusterim = ones(prm.nsubj,1);
    clustval = 1;
    
    % Cluster by kmeans
    [clusterim, ~] = clusterselection(X, clusterim, clustval, nclust-1);
            
    if subclust
        % Cluster by kmeans
        clustval = 2;
        [clusterim, ~] = clusterselection(X, clusterim, clustval, nclust-1);
        
    end
    
    % Compute centroids
    centroid = zeros(nclust, prm.nvars);
    imvar = table2array(imagevar);
    for i = 1 : nclust
        centroid(i,:) = mean(imvar(clusterim == i,:), 1);
    end
            
    % Saving cluster
    msg = ['Saving ' pathcluster];
    disp(msg);
    save(pathcluster, 'clusterim', 'centroid','preprocess');

elseif isequal(clusteroption, 'applycluster')
    
    % The variables
    vars = table2array(imagevar);
    
    % Compute distances to centroids
    dist = pdist2(vars, centroid, 'euclidean');

    % Minimum distance as the cluster
    [~, clusterim] = min(dist, [], 2);    
        
end

% Y = pdist(vars,'cityblock');
% Z = linkage(Y,'centroid');
% dendrogram(Z)
% T = clusterdata(vars,'cutoff',3);

% Report number of patients
for i = 1 : nclust
   npatclust = sum(clusterim == i);
   msg = ['Number of patients in cluster ' int2str(i) ': ' int2str(npatclust)];
   disp(msg);
end 


%-------------------------------------------------
% Print subjects in clusters to text files
%-------------------------------------------------
if isequal(clusteroption, 'createcluster')
    clear val table;
    for i = 1 : nclust
        pathsave = fullfile(printresultpath, [mfilename  '-DATAOPTION-' dataoption '-METHOD-' clustermethod '-NCLUSTERS-' int2str(nclust) '-CLUSTER-' int2str(i)  '.txt']);
        msg = ['Saving ' pathsave];
        disp(msg);
        val = prm.subj(clusterim == i);        
        tab = table(val);
        msg = ['Saving list of patients cluster ' int2str(i) ' as ' pathsave];
        disp(msg);        
        writetable(tab, pathsave,'WriteVariableNames', 0);
    end
end

% Name of clusters
nclust = numel(unique(clusterim));
clustername = cell(nclust,1);
for i = 1 : nclust            
    clustername{i} = ['C' int2str(i)];
end

%-------------------------------------------------
% Find the features with the most variance
%-------------------------------------------------

% Number of combinations
ncomb = nclust*(nclust-1)/2;

% The Euclidean distance between centroids
d2 = zeros(ncomb, prm.nvars);
iter = 0;
for i = 1 : nclust
    for j = i+1 : nclust
        if i == j
            continue;
        end
        iter = iter + 1;
        d2(iter, :) = (centroid(i,:) - centroid(j,:)).^2;
    end
end
d = sqrt(sum(d2,1));
[sortval, indsort] = sort(d, 'descend');
impfeat = prm.vars(indsort);
x = 1:prm.nvars;
fig = figure('Position', [50,100, 1200, 380]);
plot(x,sortval,'o');
xlabel('Radiomic feature ', 'FontSize', fs);
ylabel('Euclidean centroid interdistance', 'FontSize', fs);
xticks(x);
xticklabels(impfeat);
xtickangle(90);
basename = fullfile(printfigspath, [mfilename  '-DATAOPTION-' dataoption '-METHOD-' clustermethod '-NCLUSTERS-' int2str(nclust) '-ImpFeatEuclid']);
printfig(fig,basename);


% Distance between centroids
d = sqrt(d2);
x = 1:prm.nvars;
fig = figure('Position', [50,100, 1200, 380]);
plot(x,d(:,indsort),'o-', 'LineWidth', 1);
xlabel('Radiomic feature ', 'FontSize', fs);
ylabel('Inter-centroide distance (a.u.)', 'FontSize', fs);
xticks(x);
xticklabels(prm.vars(indsort));
xtickangle(90);
legend('Cluster 1 vs. Cluster 2','Cluster 1 vs. Cluster 3','Cluster 2 vs. Cluster 3', 'FontSize', fs);
basename = fullfile(printfigspath, [mfilename  '-DATAOPTION-' dataoption '-METHOD-' clustermethod '-NCLUSTERS-' int2str(nclust) '-ImpFeat']);
printfig(fig,basename);

%------------------------------------------
% Plot heatmap
%------------------------------------------

fs = 10;
% Sort patients according to cluster
[sortval, sortind] = sort(clusterim);
var = imagevar.Variables';
var = var(:,sortind);
bot1 = 0.2;
b = 0.01;
fig = figure('Position',  [100, 100, 750, 750]);
sfig1 = subplot('Position', [0.2,bot1,0.75,0.75]);
[f, hText, hTick] = heatmap(var, [], prm.vars, [], 'TickAngle', 45,...
        'ShowAllTicks', true, 'TickFontSize', 12, 'Colorbar', true, 'Parent', sfig1);
% hold on;
set(gca, 'FontSize', fs);

caxis([-6,8]);
% plot([0.5, 0.5], [-0.5, prm.nvars+0.5],'-k', 'LineWidth', 2)
hold on;
if nclust == 2
    mid = sum(clusterim == 1);   
    plot([mid, mid], [-0.5, prm.nvars+0.5],'-k', 'LineWidth', 2)
    an = annotation('doublearrow');
    set(an, 'X', [0.215,0.525], 'Units', 'Normalized')
    set(an, 'Y', [0.09,0.09], 'Units', 'Normalized')
    an = annotation('doublearrow');
    set(an, 'X', [0.525,0.845], 'Units', 'Normalized')
    set(an, 'Y', [0.09,0.09], 'Units', 'Normalized')
    str = 'Cluster 1';
    annotation('textbox',[.32 0.075 .1 .01],'String',str,'FitBoxToText','on','LineStyle','None','FontSize',fs);
    str = 'Cluster 2';
    annotation('textbox',[.64 0.075 .1 .01],'String',str,'FitBoxToText','on','LineStyle','None','FontSize',fs);
    
elseif nclust == 3
    mid1 = sum(clusterim == 1);    
    hold on    
    plot([mid1, mid1], [-0.5, prm.nvars+0.5],'-k', 'LineWidth', 2)
    mid2 = mid1 + sum(clusterim == 2);
    hold on    
    plot([mid2, mid2], [-0.5, prm.nvars+0.5],'-k', 'LineWidth', 2)
    an = annotation('doublearrow');
    if isequal(dataoption, 'manual')
        set(an, 'X', [0.20,0.55], 'Units', 'Normalized')
        set(an, 'Y', [bot1,bot1]-b, 'Units', 'Normalized')
        an = annotation('doublearrow');
        set(an, 'X', [0.55,0.77], 'Units', 'Normalized')
        set(an, 'Y', [bot1,bot1]-b, 'Units', 'Normalized')
        an = annotation('doublearrow');
        set(an, 'X', [0.77,0.89], 'Units', 'Normalized')
        set(an, 'Y', [bot1,bot1]-b, 'Units', 'Normalized')
        str = 'Cluster 1';
        annotation('textbox',[.32 bot1-2*b .1 .01],'String',str,'FitBoxToText','on','LineStyle','None','FontSize',fs);
        str = 'Cluster 2';
        annotation('textbox',[.62 bot1-2*b .1 .01],'String',str,'FitBoxToText','on','LineStyle','None','FontSize',fs);
        str = 'Cluster 3';
        annotation('textbox',[.79 bot1-2*b .1 .01],'String',str,'FitBoxToText','on','LineStyle','None','FontSize',fs);
    elseif isequal(dataoption, 'ML')
        l1 = 0.56;
        l2 = 0.74;
        set(an, 'X', [0.20,l1], 'Units', 'Normalized')
        set(an, 'Y', [bot1,bot1]-b, 'Units', 'Normalized')
        an = annotation('doublearrow');        
        set(an, 'X', [l1,l2], 'Units', 'Normalized')
        set(an, 'Y', [bot1,bot1]-b, 'Units', 'Normalized')
        an = annotation('doublearrow');
        set(an, 'X', [l2,0.89], 'Units', 'Normalized')
        set(an, 'Y', [bot1,bot1]-b, 'Units', 'Normalized')
        str = 'Cluster 1';
        annotation('textbox',[.32 bot1-2*b .1 .01],'String',str,'FitBoxToText','on','LineStyle','None','FontSize',fs);
        str = 'Cluster 2';
        annotation('textbox',[.62 bot1-2*b .1 .01],'String',str,'FitBoxToText','on','LineStyle','None','FontSize',fs);
        str = 'Cluster 3';
        annotation('textbox',[.79 bot1-2*b .1 .01],'String',str,'FitBoxToText','on','LineStyle','None','FontSize',fs);        
    end
end

% Save figure
basename =  fullfile(printfigspath, [mfilename '-HeatMap' '-DATAOPTION-' dataoption '-CLUSTEROPTION-' clusteroption  '-NCLUSTERS-' int2str(nclust)]);
printfig(fig,basename);


