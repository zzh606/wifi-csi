% Generate Doppler Spectrum
[doppler_spectrum, freq_bin] = get_doppler_spectrum([dpth_ges, spfx_ges],...
                    rx_cnt, rx_acnt, 'stft');
%  save('./dd.mat','doppler_spectrum','freq_bin')               
% Doppler To Velocity Mapping [Nonlinear Programming]
% Target: doppler_spectrum_seg: 2*F*N; doppler_spectrum_seg: 2*F
%        F:frequency bin number; N: segment length
% Approximation: velocity_spectrum_seg: M*M
%        M: velocity bin number

% Segment Settings
ges_per_file = 1;   
norm = 0;           
lambda = 1e-7;      
torso_pos = [1.365 0.455; 0.455 0.455; 0.455 1.365; 1.365 1.365; 0.91 0.91;...  % pos1-5
    2.275 1.365; 2.275 2.275; 1.365 2.275];  %	pos6-8      
torso_ori = [-90, -45, 0, 45, 90];  
Tx_pos = [0 0];
Rx_pos = [0.455 -0.455; 1.365 -0.455; 2.0 0;... % Rx1-3
    -0.455 0.455; -0.455 1.365; 0 2.0]; % Rx4-6     
seg_length = 100;   
wave_length = 299792458 / 5.825e9;
V_max = 2;
V_min = -2;
V_bins = 20;    
V_resolution = (V_max - V_min)/V_bins;
M = (V_max - V_min)/V_resolution;
velocity_bin = ((1:M) - M/2) / (M/2) * V_max;
MaxFunctionEvaluations = 100000;    

% Cyclic Doppler Spectrum According To frequency bin
[~,idx] = max(freq_bin);
circ_len = length(freq_bin) - idx;
doppler_spectrum = circshift(doppler_spectrum, [0 circ_len 0]);

% plot(1:size(doppler_spectrum,2),doppler_spectrum(1,:,10))

% for kk = 1:6
%     figure;
%     
%     colormap(jet);
%     mesh(1:size(doppler_spectrum,3),-60:60,squeeze(doppler_spectrum(kk,:,:)));view([0,90]);
%     xlim([0,size(doppler_spectrum,3)]);ylim([-60,60]);
%     set(gcf,'WindowStyle','normal','Position', [300,300,400,250]); % window size
%     set (gca,'color','none', 'fontsize', 12); % fontsize
%     set(gca,'yTick',-60:20:60);
%     xlabel('Time (ms)'); % x label
%     ylabel('Frequency (Hz)'); % y label
% 
%     colorbar; %Use colorbar only if necessary
%     caxis([min(doppler_spectrum(:)),max(doppler_spectrum(:))]);
% end

% For Each Segment Do Mapping
doppler_spectrum_max = max(max(max(doppler_spectrum,[],2),[],3));
U_bound = repmat(doppler_spectrum_max, M, M);
A = get_A_matrix(torso_pos(pos_sel,:), Tx_pos, Rx_pos, rx_cnt);
VDM = permute(get_velocity2doppler_mapping_matrix(A, wave_length,...
    velocity_bin, freq_bin, rx_cnt), [2,3,1,4]);    % 20*20*rx_cnt*121
CastM = get_CastM_matrix(A, wave_length, velocity_bin, freq_bin);

for ges_number = 1:1 %ges_per_file
    seg_number = floor(size(doppler_spectrum, 3)/seg_length);
    doppler_spectrum_ges = doppler_spectrum;
    velocity_spectrum = zeros(M, M, seg_number);
    parfor ii = 1:1 %seg_number
        % Set-up fmincon Input
        doppler_spectrum_seg = doppler_spectrum_ges(:,:,...
            (ii - 1)*seg_length+1 : ii*seg_length);
        doppler_spectrum_seg_tgt = mean(doppler_spectrum_seg, 3);
        
        % Normalization Between Receivers(Compensate Path-Loss)
        for jj = 2:size(doppler_spectrum_seg_tgt,1)
            if any(doppler_spectrum_seg_tgt(jj,:))
                doppler_spectrum_seg_tgt(jj,:) = doppler_spectrum_seg_tgt(jj,:)...
                    * sum(doppler_spectrum_seg_tgt(1,:))/sum(doppler_spectrum_seg_tgt(jj,:));
            end
        end

        % Apply fmincon Solver
        [P,fval,exitFlag,output] = fmincon(... % @(P)DVM_target_func是关于P的函数表达式
            @(P)DVM_target_func(P, VDM, lambda, doppler_spectrum_seg_tgt, size(doppler_spectrum_seg_tgt,1), norm),...
            zeros(M, M),...  % Initial Value，P的初始值
            [],[],...       % Linear Inequality Constraints
            [],[],...       % Linear Equality Constraints
            zeros(M,M),...  % Lower Bound
            U_bound,...     % Upper Bound 
            @(P)DVM_nonlinear_func(P, CastM),... % [],...    % Non-linear Constraints
            optimoptions('fmincon','Algorithm','sqp',...
            'MaxFunctionEvaluations', MaxFunctionEvaluations));	% Options
        velocity_spectrum(:,:,ii) = P;
        exitFlag
    end
    
    % Rotate Velocity Spectrum According to Orientation
    velocity_spectrum_ro = get_rotated_spectrum(velocity_spectrum, torso_ori(ori_sel));
    
    % Save VS
    save([dpth_vs, dpth_people, '-', spfx_ges, '-', num2str(ges_number),...
        '-', num2str(lambda), '-', num2str(seg_length),...
        '-', num2str(V_bins), '-', num2str(MaxFunctionEvaluations),...
        '-L', num2str(norm), '.mat'], 'velocity_spectrum_ro');
end




