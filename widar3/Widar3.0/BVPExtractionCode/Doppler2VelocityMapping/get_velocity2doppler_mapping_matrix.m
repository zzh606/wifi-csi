% Input:(If rx_cnt=2)
% A: 2*2
% velocity_bin: 1*M
% freq_bin: 1*F
% Output: 2*M*M*F, component is ...
function VDM = get_velocity2doppler_mapping_matrix(A, wave_length,...
        velocity_bin, freq_bin, rx_cnt)
    if size(A,1) ~= rx_cnt
        error('Error Rx Count!');
    end
    F = size(freq_bin, 2);
    M = size(velocity_bin, 2);
    freq_min = min(freq_bin);
    freq_max = max(freq_bin);
    
    VDM = zeros(rx_cnt, M, M, F);
    % For Each Link
    for ii = 1:rx_cnt
        for i = 1:M
            for j = 1:M
                plcr_hz = round(A(ii,:) * [velocity_bin(i) velocity_bin(j)]'...
                    / wave_length);
                if plcr_hz > freq_max || plcr_hz < freq_min
                    VDM(ii,i,j,:) = 1e10*ones(1,size(VDM,4));
                    continue;
                end
                idx = plcr_hz + 1 - freq_min;
                VDM(ii,i,j,idx) = 1;
            end
        end
    end
end