% SHOWUD Shows image planes where it is possible to go up and down in the
% stack.
% SHOWUD(I) shows the image planes of I and it is possible to go up and
% down in the stack. Starts at plane 1
% SHOWUD(I,FIGNUM) shows the image planes in figure number FIGNUM 
% of I and it is possible to go up and down in the stack. Starts at plane 1
%
function [] = showud(varargin)

if nargin == 1
    figNum = 1;
elseif nargin > 2    
    error('Wrong number of inputs');
end;

global plane cont;

I = varargin{1};
    
[M N O] = size(I);

plane = 1;
figure(figNum);imagesc(I(:,:,plane));colormap(gray);title(sprintf('Plane %i',plane));
h2 = uicontrol('style','pushbutton','Units','normalized','Position',...
    [0.91,0.1,0.09,0.09],'String','Down',...
    'CallBack','global plane cont; plane = plane - 1;cont = 1;');
h2 = uicontrol('style','pushbutton','Units','normalized','Position',...
    [0.91,0.2,0.09,0.09],'String','Up',...
    'CallBack','global plane cont; plane = plane + 1;cont = 1;');
while 1
    
    if plane < 1
        plane = 1;
    elseif plane > O
        plane = O;
    end;
        
    if cont == 0
        pause(0.1);
    else
        disp(sprintf('Plane %i',plane))
        figure(figNum);imagesc(I(:,:,plane));colormap(gray);title(sprintf('Plane %i',plane));
        cont = 0;        
    end;
    
end;

