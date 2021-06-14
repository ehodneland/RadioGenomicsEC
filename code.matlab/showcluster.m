close all;
settingsgeneticdata;

D = load('../ManuscriptGeneAnalysis/results/analysegeneticdata-NearCentroid.mat');
avim = D.avim;

prepathMR = '/raid/erlend/GynKreft/Data-EC/Nifti';


% Load patient data
% imagevar = readtable(pathrawimage);
% listvibe = imagevar.listvibe;
% subjMR = imagevar.subject;

nsubj = size(avim.subj,1);
nclust = size(avim.subj,2);
vol = NaN(nsubj, nclust);
A = cell(nsubj, 2);
for i = 1 : nclust
    msg = ['Cluster ' int2str(i)];
    disp(msg);
    for j = 1 : nsubj
        subj = avim.subj(j,i);        
        subj = makezero(3, subj);
        
        % Load image data
        pathload = fullfile(prepathMR, subj, 'unregistered', 'vibe2min.nii.gz');        
        msg = ['Loading ' pathload];
        disp(msg);
        [im, hdr] = cbiReadNifti(pathload);
        h = hdr.pixdim(2:4)';
        dim = size(im);
        
        if i == 1
            if j == 1                
                href = h;
            end
        end                  

        % Load segmentation
        try
            pathload = fullfile(prepathMR, subj, 'registered', [subj 'segmentedJulie-2-vibe2min-header.nii.gz']);
            [segmbw, hdr] = cbiReadNifti(pathload);
        catch
            pathload = fullfile(prepathMR, subj, 'segmented', [subj 'segmentedKari.nii.gz']);
            [segmbw, hdr] = cbiReadNifti(pathload);            
        end        
        msg = ['Loading ' pathload];
        disp(msg);
        
        % Volume in ml
        vol(j,i) = sum(segmbw(:))*prod(h)/1000;

        if ~isequal(h, href)
           ratio = h./href;
           dimnew = round(dim.*ratio);
           im = imresize3d(im, dimnew, 'bilinear');
           segmbw = imresize3d(segmbw, dimnew, 'nearest');
           h = h./ratio;
        end
        
        % Dimension image
        dim = size(segmbw);        
        msg = ['Dimension: ' int2str(dim)];
        disp(msg);
        
        % Find the slice with maximum area of tumor
        ind = find(segmbw);
        clear c;        
        [c(:,1), c(:,2), c(:,3)] = ind2sub(dim, ind);
        val = c(:,3);        
        [slice,~] = mostfrequent(val);
        msg = ['Plotting slice ' int2str(slice)];
        disp(msg);
        
        % Take out the slice we want to plot
        im = im(:,:,slice);
        segmbw = segmbw(:,:,slice);
        
        % Make a clip outside the tumor
        clear c;
        [c(:,1), c(:,2), c(:,3)] = ind2sub(dim, find(segmbw));
        meanc = round(mean(c, 1));
        
        p = 40;
        im = im(meanc(1)-p:meanc(1)+p,meanc(2)-p:meanc(2)+p);
        segmbw = segmbw(meanc(1)-p:meanc(1)+p,meanc(2)-p:meanc(2)+p);
        
        % Squeeze and rotate
        im = squeeze(imoverlayrgb(im, segmbw, 1));
        im = imrotate(im, 90);
        A{j,i} = im;
    end
end
meanvol = mean(vol,2);
[sortval, sortind] = sort(meanvol);
A = A(sortind, :);
H = panelstruct(A, 0.002, 800);
basename = fullfile(printfigspath, [mfilename, 'AveragePatients']);
printfig(H,basename)

msg = ['Tumor volumes in 3D:'];
disp(msg);
vol

msg = ['FIGO stage:'];
disp(msg);
avim.FIGO

msg = ['Histologic type:'];
disp(msg);
avim.HistType

msg = ['Histologic grade:'];
disp(msg);
avim.HistGrade



