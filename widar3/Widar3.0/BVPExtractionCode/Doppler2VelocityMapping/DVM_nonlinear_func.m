% Input:
% P: M*M
% CastM: M*M
% Non-linear Constraints
function [fc, fceq] =  DVM_nonlinear_func(P, CastM)
    fc = [];    % None Non-linear Inequality Constraints
    
    fceq = sum(sum(P .* CastM));
end