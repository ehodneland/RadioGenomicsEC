function [vars, means, stds] = normalizevars(vars, bw, means, stds)

n = sum(bw);

if isempty(means)
    means = mean(vars(bw,:), 1);    
end
vars(bw, :) = vars(bw,:)-repmat(means, n, 1);

if isempty(stds)
    stds = std(vars(bw, :), [], 1);    
end
vars(bw,:) = vars(bw,:)./repmat(stds, n, 1);
