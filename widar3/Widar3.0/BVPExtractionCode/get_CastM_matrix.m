% Cast Matrix For Non-linear Inequality Constraints
% Making Sure That VelocitySpectrum Is Within Doppler Bound
function CastM = get_CastM_matrix(A, wave_length, velocity_bin, freq_bin)
    A = A(1:2,:);
    M = size(velocity_bin, 2);
    F_max = max(freq_bin);
    F_min = min(freq_bin);
    CastM = zeros(M, M);
    
    for ii = 1:M
        for jj = 1:M
            plcr_hz = round(A * [velocity_bin(ii) velocity_bin(jj)]' / wave_length);
%             if max(plcr_hz) > F_max || min(plcr_hz) < F_min
%                 CastM(ii, jj) = 1;
%             end
            if sqrt(sum(plcr_hz.^2)) > F_max
                CastM(ii, jj) = 1;
            end
        end
    end
end