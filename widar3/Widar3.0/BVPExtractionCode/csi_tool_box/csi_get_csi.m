function [ret, time_stamp, cfr_a_array, cfr_p_array] = csi_get_csi(filename,antenna)
usePerm = 0;
%csi_trace = read_bf_file(strcat('csi_nlos_5',num2str(i),'.dat'));
csi_trace = read_bf_file(filename);
time_stamp = zeros(length(csi_trace), 1);
cfr_a_array = zeros(length(csi_trace),30); % Amplitude of CFR for all packets
cfr_p_array = zeros(length(csi_trace),30); % Phase of CFR for all packets
%cir_a_array = zeros(length(csi_trace),nfft); % Amplitude of CIR for uall packets
%cir_p_array = zeros(length(csi_trace),nfft); % Phase of CIR for all packets
cfr_array = zeros(length(csi_trace), 30);

for k = 1:length(csi_trace)
    csi_entry = csi_trace{k}; % for the k_{th} packet
    
    csi_all = squeeze(get_scaled_csi(csi_entry)).'; % estimate channel matrix Hexp-figu
    csi = csi_all(:,antenna); % select CSI for one antenna pair
    
    cfr = csi;
    cfr_a_array(k,:) = abs(cfr);
    cfr_p_array(k,:) = angle(cfr);
    time_stamp(k) = csi_entry.timestamp_low;
    cfr_array(k,:) = cfr;
end

ret = cfr_array;