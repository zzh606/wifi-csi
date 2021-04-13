clear;
clc;
close all;

addpath('.\csi_tool_box')
addpath('.\Data')
addpath(genpath('.\tftb'));


start_index = [1 1 1 1];
total_mo = 1;   % Total motion count
total_pos = 1;  % Total position count
total_ori = 1;  % Total orientation count
total_ges = 1;  % Total gesture repeatation count
start_index_met = 0;
rx_cnt = 6;     % Receiver count(no less than 3)
rx_acnt = 3;    % Antenna count for each receiver
dpth_pwd = './';
dpth_date = 'Data';
dpth_people = 'userA';
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dpth_ges = [dpth_pwd, dpth_date, '/'];
dpth_vs = [dpth_pwd, 'BVP/'];

tic
for mo_sel = 1:total_mo
    for pos_sel = 1:total_pos
        for ori_sel = 1:total_ori
            for ges_sel = 1:total_ges
                spfx_ges = [dpth_people, '-', num2str(mo_sel), '-', num2str(pos_sel),...
                    '-', num2str(ori_sel), '-', num2str(ges_sel)];
                if mo_sel == start_index(1) && pos_sel == start_index(2) &&...
                        ori_sel == start_index(3) && ges_sel == start_index(4)
                    start_index_met = 1;
                end
                if start_index_met == 1
                    disp(['Running ', spfx_ges])
                    try
                        [doppler_spectrum, freq_bin] = get_doppler_spectrum(...
                            spfx_ges, rx_cnt, rx_acnt, 'stft');
                    catch err
                        disp(['Exception Occured' err.message]);
                    	continue;
                    end
                else
                    disp(['Skipping ', spfx_ges])
                end
            end
        end
    end
end
toc
