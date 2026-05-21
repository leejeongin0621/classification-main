clear; close all; clc

path_cur = cd();

sess = 5;
class = 100;

subject_name = '250806_LJI';
% 예시:
% '250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM',
% '250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB'
% '251223_KSH','251229_JJH','251230_LSW','251230_LJS','260126_KCY'

axis = 3;
num_sensor = 10;
total_ch = 3*num_sensor;

epo.fs = 50;    % 100Hz -> 50Hz
total_t = 5;
t = 4;

disp(['Start preprocessing: ', num2str(epo.fs), ' Hz']);
disp(['Subject: ', subject_name]);

path_load = fullfile(path_cur, ['rawdata_' num2str(num_sensor) 'ch']);
path_save = fullfile(path_cur, ['data_' num2str(num_sensor) 'ch']);

for i_sess = 1:sess

    epo.x = zeros(epo.fs*t, total_ch, class);
    epo.desc = 1:class;
    epo.y = onehotencode(categorical(epo.desc), 1);

    for i_data = 1:class

        path_load_sess = fullfile(path_load, subject_name, ['session_' num2str(i_sess)]);
        cd(path_load_sess)

        DataName = strcat('Subject:', subject_name, ...
                          ' Session:', num2str(i_sess), ...
                          ' Word:', num2str(i_data));
        disp(DataName)

        load(strcat('sentence_', num2str(i_sess), '_', num2str(i_data), '_Accel_data.mat'));
        load(strcat('sentence_', num2str(i_sess), '_', num2str(i_data), '_Accel_data2.mat'));

        target_len = epo.fs * total_t;

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

        rs_accel  = rs_accel(0.5*epo.fs + 1 : epo.fs*t + 0.5*epo.fs, :);
        rs_accel2 = rs_accel2(0.5*epo.fs + 1 : epo.fs*t + 0.5*epo.fs, :);

        accel_all = cat(2, rs_accel, rs_accel2);
        epo.x(:, :, i_data) = accel_all;

        path_save_sess = fullfile(path_save, subject_name);

        if ~exist(path_save_sess, 'dir')
            mkdir(path_save_sess)
        end

        save(fullfile(path_save_sess, ['epo_session' num2str(i_sess) '.mat']), 'epo');
    end
end

plot_sensor = 7;        % sensor 번호: 1~10
plot_word = 99;          % word 번호: 1~100

figure('Position', [100 50 1400 900]);

for plot_sess = 1:sess

    path_plot = fullfile(path_save, subject_name);
    load(fullfile(path_plot, ['epo_session' num2str(plot_sess) '.mat']));

    x = epo.x;   % (time, channel, class)

    ch_x = 3*(plot_sensor-1) + 1;
    ch_y = 3*(plot_sensor-1) + 2;
    ch_z = 3*(plot_sensor-1) + 3;

    % x axis
    subplot(sess, 3, 3*(plot_sess-1)+1);
    plot(x(:, ch_x, plot_word), 'LineWidth', 1.2);
    grid on;
    ylabel(['Sess ' num2str(plot_sess)]);
    if plot_sess == 1
        title('x axis');
    end

    % y axis
    subplot(sess, 3, 3*(plot_sess-1)+2);
    plot(x(:, ch_y, plot_word), 'LineWidth', 1.2);
    grid on;
    if plot_sess == 1
        title('y axis');
    end

    % z axis
    subplot(sess, 3, 3*(plot_sess-1)+3);
    plot(x(:, ch_z, plot_word), 'LineWidth', 1.2);
    grid on;
    if plot_sess == 1
        title('z axis');
    end

end

sgtitle([subject_name ...
    ' / Sensor ' num2str(plot_sensor) ...
    ' / Word ' num2str(plot_word) ...
    ' / All Sessions']);