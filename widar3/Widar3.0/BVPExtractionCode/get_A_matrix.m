function A = get_A_matrix(torso_pos, Tx_pos, Rx_pos, rxcnt)
    if rxcnt > size(Rx_pos,1)
        error('Error Rx Count!')
    end
    A = zeros(rxcnt,2);

    for ii = 1:rxcnt
        dis_torso_tx = sqrt((torso_pos-Tx_pos) * (torso_pos-Tx_pos)');
        dis_torso_rx = sqrt((torso_pos-Rx_pos(ii,:)) * (torso_pos-Rx_pos(ii,:))');
        A(ii,:) = (torso_pos - Tx_pos)/dis_torso_tx + ...
            (torso_pos - Rx_pos(ii,:))/dis_torso_rx;
    end
end