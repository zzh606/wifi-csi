% Input:
% P: M*M
% VDM: 2*M*M*F
% doppler_spectrum_seg_tgt: 2*F
function floss = DVM_target_func(P,...
    VDM, lambda, doppler_spectrum_seg_tgt, rx_cnt, norm)
    if(size(P,1) ~= size(P,2))
        return;
    end
    % Initialize Variable
    M = size(P,1);
    F = size(doppler_spectrum_seg_tgt,2);
    P_extent = repmat(P, 1, 1, rx_cnt, F);
    
    % Construct Approximation Doppler Spectrum
    doppler_spectrum_seg_approximate = squeeze(sum(squeeze(sum(P_extent .* VDM, 1)), 1));

    %%% Construct Loss Function %%%
    % EMD Distance
    floss = 0;
    for ii = 1:rx_cnt
        if(any(doppler_spectrum_seg_tgt(ii,:)))
            floss = floss + sum(abs((doppler_spectrum_seg_approximate(ii,:)...
                - doppler_spectrum_seg_tgt(ii,:)) * triu(ones(F,F),0)));
        end
    end
    % Norm Loss
    if norm == 1
        floss = floss + lambda * sum(sum(P));
    else if norm == 0
            floss = floss + lambda * sum(sum(P~=0));
        end
    end
%     floss
end
