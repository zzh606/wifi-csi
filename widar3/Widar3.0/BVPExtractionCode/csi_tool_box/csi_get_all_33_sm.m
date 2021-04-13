function [cfr_array, timestamp] = csi_get_all_33_sm(filename)
csi_trace = read_bf_file(filename);
timestamp = zeros(length(csi_trace), 1);
cfr_array = zeros(length(csi_trace), 270);

for k = 1:length(csi_trace)
    csi_entry = csi_trace{k}; % for the k_{th} packet
    
    csi_all = squeeze(get_scaled_csi_sm(csi_entry)); % estimate channel matrix Hexp-figu
    csi = [squeeze(csi_all(1,1,:))' squeeze(csi_all(1,2,:))' squeeze(csi_all(1,3,:))' ...
        squeeze(csi_all(2,1,:))' squeeze(csi_all(2,2,:))' squeeze(csi_all(2,3,:))' ...
        squeeze(csi_all(3,1,:))' squeeze(csi_all(3,2,:))' squeeze(csi_all(3,3,:))']; % select CSI for one antenna pair
    
    timestamp(k) = csi_entry.timestamp_low;
    cfr_array(k,:) = csi;
end