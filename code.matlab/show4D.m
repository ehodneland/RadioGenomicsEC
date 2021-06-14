function [ ] = show4D( varargin )
%  SHOW4D Show 4D data
%
%   SHOW4D(IM) Show 4D data IM inteartively
%
%     =======================================================================================
%     Copyright (C) 2013  Erlend Hodneland
%     Email: erlend.hodneland@biomed.uib.no 
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%     =======================================================================================
%


    global choice
    choice = 0;
    
    cim = 0;
    cnargin = 0;

    while 1
        cnargin = cnargin + 1;    

        if isequal(varargin{cnargin},'clim')
            clim = varargin{cnargin+1};
            cnargin = cnargin + 1;
        else
            cim = cim + 1;
            f{cim} = varargin{cnargin};
            H(cim) = figure(cim);
            dimhere = size(f{cim});
            try
                dim(4) = max(dim(4),dimhere(4));
            catch
            end        
        end
        if cnargin == nargin
            break;
        end
    end
    nim = cim;

    dim = ones(1,4);
    a = size(f{1});
    n = numel(a);
    dim(1:n) = a;

    prm.plane = round(dim(3)/2);
    prm.time = 1;
    prm.cont = 0;

    dia = figure('Position',[50 500 200 150],'Name','Select One');
    l = 20;
    w = 150;
    h = 20;
    b = 125;
    up = uicontrol('Parent',dia,...           
           'Position',[l b w h],...
           'String','Up',...
           'Callback',@upfcn);       
    s = 10;
    b=b-h-s;
    down = uicontrol('Parent',dia,...           
           'Position',[l b w h],...
           'String','Down',...
           'Callback',@downfcn);
    b=b-h-s;
    previous = uicontrol('Parent',dia,...           
           'Position',[l b w h],...
           'String','Previous',...
           'Callback',@previousfcn);
    b=b-h-s;
    next = uicontrol('Parent',dia,...           
           'Position',[l b w h],...
           'String','Next',...
           'Callback',@nextfcn);
    b=b-h-s;   
    qt = uicontrol('Parent',dia,...
           'Position',[l b w h],...
           'String','Close',...
           'Callback',@clfcn);
          
    while 1
        prm.plane = min(prm.plane,dim(3));
        prm.plane = max(prm.plane,1);
        prm.time = min(prm.time,dim(4));
        prm.time = max(prm.time,1);
        msg = ['Plane: ' int2str(prm.plane) ', time: ' int2str(prm.time)];
        disp(msg);

        for i = 1 : nim        
            figure(H(i));
            if exist('clim','var')
                try
                    imagesc(f{i}(:,:,prm.plane,prm.time),clim);colormap(gray);axis image;
                catch
                    imagesc(f{i}(:,:,prm.plane),clim);colormap(gray);axis image;
                end
            else
                try
                    imagesc(f{i}(:,:,prm.plane,prm.time));colormap(gray);axis image;
                catch
                    imagesc(f{i}(:,:,prm.plane));colormap(gray);axis image;
                end
            end
        end

        % Wait for d to close before running to completion
        uiwait(dia);
        
        if choice == 1
            prm.plane = prm.plane + 1;
        elseif choice == 2
            prm.plane = prm.plane - 1;
        elseif choice == 3
            prm.time = prm.time - 1;
        elseif choice == 4
            prm.time = prm.time + 1;
        elseif choice == 5
            close all;
            delete(dia)
            return;
        end
        choice = 0;
        % reset
%         prm.cont = 0;
    end
%     close(dia);
    function upfcn(popup,event)
        choice = 1; 
        uiresume;
    end    
    function downfcn(popup,event)
        choice = 2;
        uiresume;
    end    
    function previousfcn(popup,event)
        choice = 3;
        uiresume;
    end    
    function nextfcn(popup,event)
        choice = 4;
        uiresume;
    end    
    function clfcn(popup,event)
        choice = 5;
        uiresume;
    end    
end



