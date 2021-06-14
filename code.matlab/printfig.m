function [] = printfig(H,basename)

H.PaperPositionMode = 'auto';
pathsave = [basename, '.eps'];
msg = ['Saving ' pathsave];
disp(msg);
print(pathsave, '-depsc');
pathsave = [basename, '.png'];
msg = ['Saving ' pathsave];
disp(msg);
print(pathsave, '-dpng');
    

