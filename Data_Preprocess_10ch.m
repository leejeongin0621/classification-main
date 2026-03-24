clear; close all; clc

path_cur = cd();

sess = 5;
class = 100;
sub = 'KTH';
% LGE, KTS, KDY, SMC, LJI, LSC(HM), JJH, LPR, JHS, HHJ, WDY, YMS, JCM,
% CYJ,ICY, CYK, PYH, LTG, JSH, KTH
date = 250822;
% 250731, 250804, 250805,250805, 250806, 250808,250808, 250811, 250811,
% 250812,250812, 250813, 250814,250814, 250818, 250819, 250819, 250820
% 250822, 250822
axis = 3;
num_sensor = 10;
total_ch = 3*num_sensor;

epo.fs = 50;    % 100Hz -> 50Hz
total_t = 5;
t = 4;
disp(['Start preprocessing:', num2str(epo.fs), ' Hz']);
path_load = [path_cur, '\rawdata_', num2str(num_sensor), 'ch'];
path_save = [path_cur, '\data_', num2str(num_sensor), 'ch_', num2str(epo.fs), 'Hz'];


for i_sess = 1:sess

    epo.x = zeros(epo.fs*t, total_ch, class);
    epo.desc = 1:class;
    epo.y = onehotencode(categorical(epo.desc),1);

    for i_data = 1:class

        path_load_sess = fullfile(path_load, [num2str(date) '_' sub], ['session_' num2str(i_sess)]);
        cd(path_load_sess)

        DataName = strcat('Subject:',num2str(sub),' Session:',num2str(i_sess), ' Word:',num2str(i_data));
        disp(DataName)

        load(strcat('sentence_', num2str(i_sess), '_',num2str(i_data),'_Accel_data.mat'));
        load(strcat('sentence_', num2str(i_sess), '_',num2str(i_data),'_Accel_data2.mat'));

        target_len = epo.fs*total_t;
        [n_sensor, n_axis, orig_len] = size(savedata_accel);
        xi = linspace(1, orig_len, target_len);
        rs_accel = zeros(target_len, total_ch/2);
        for s = 1:n_sensor
            for ax = 1:axis
                sig = squeeze(savedata_accel(s, ax, :)) / 16384;
                
                rs_accel(:, 3*(s-1)+ax) = interp1(1:orig_len, sig, xi, 'pchip');
            end
        end

        [n_sensor, n_axis, orig_len] = size(savedata_accel2);
        xi = linspace(1, orig_len, target_len);
        rs_accel2 = zeros(target_len, total_ch/2);
        for s = 1:n_sensor
            for ax = 1:axis
                sig = squeeze(savedata_accel2(s, ax, :)) / 16384;
                rs_accel2(:, 3*(s-1)+ax) = interp1(1:orig_len, sig, xi, 'pchip');
            end
        end
        

        rs_accel = rs_accel(0.5*epo.fs + 1:epo.fs*t+0.5*epo.fs, :);
        rs_accel2 = rs_accel2(0.5*epo.fs+ 1:epo.fs*t+0.5*epo.fs, :);

        accel_all = cat(2, rs_accel, rs_accel2);
        epo.x(:,:,i_data) = accel_all;

        path_save_sess = fullfile(path_save, [num2str(date) '_' sub]);
        if ~ exist(path_save_sess)
            mkdir(path_save_sess)
        end
        save(strcat(path_save_sess, '\epo_session', num2str(i_sess)), 'epo');
    end

end


% 그냥 가운데 400 자르기...?