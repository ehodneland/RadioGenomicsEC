% SHOW Drawing images using colormap gray and imagesc function.
%
%   SHOW(IM) is drawing the image in IM at the middle plane
%
%   SHOW(IM,FIGNUM) is drawing the image in IM at the middle plane in the
%   figure FIGNUM
%
function [] = show(varargin)

global inactiv    

I = varargin{1};
fignum = [];
if nargin >= 2
    fignum = varargin{2};
end

h = [1,1];
if nargin >= 3
    h = varargin{3};
end;

if issparse(I);
    I = full(I);
end;

dim = size(I);
if numel(dim) == 2
    dim = [dim,1];
end;
middle = round(dim(3)/2);
a = [h(1)];
b = [h(2)];

if isempty(fignum)
    figure;
else
    figure(fignum);
end;            
colormap(gray);imagesc(b,a,I(:,:,middle));axis image;drawnow


