clear; close all; clc

baseDir = "data_10ch";

subs  = ["LGE","KTS","KDY","SMC","LJI","LSC","JJH","LPR","JHS","HHJ","WDY","YMS","JCM","CYJ","ICY","CYK","PYH","LTG","JSH","KTH"];
dates = [250731,250804,250805,250806,250808,250811,250812,250813,250814,250818,250819,250820,250822];

%folder 추가하기
pairs = strings(0,1);
for d = dates
    for s = subs
        folder = baseDir + filesep + sprintf("%d_%s", d, s);
        if isfolder(folder)
            pairs(end+1,1) = folder;
        end
    end
end

if isempty(pairs)
    error("No folders found under %s", baseDir);
end

% --- viewer state ---
i = 1;          % 현재 sub index
sess = 1;       % 현재 session (1~5)

while true
    folder = pairs(i);

    % session 파일 경로
    fpath = fullfile(folder, sprintf("epo_session%d.mat", sess));

    % 파일 없으면, 있는 session 찾아서 자동 이동
    if ~isfile(fpath)
        found = false;
        for ss = 1:5
            tmp = fullfile(folder, sprintf("epo_session%d.mat", ss));
            if isfile(tmp)
                sess = ss;
                fpath = tmp;
                found = true;
                break;
            end
        end
        if ~found
            fprintf("[SKIP] %s (no epo_session*.mat)\n", folder);
            i = i + 1;
            if i > numel(pairs), break; end
            continue;
        end
    end

    % --- load ---
    S = load(fpath);
    if ~isfield(S,"epo") || ~isfield(S.epo,"x")
        fprintf("[SKIP] %s (epo.x not found)\n", fpath);
        i = i + 1;
        if i > numel(pairs), break; end
        continue;
    end

    x = S.epo.x;  % 기대: time × channel × trial

    % --- trial 평균내기 ---
    if ndims(x) >= 3
        x_mean = mean(x, 3, "omitnan");  % time × channel
    else
        x_mean = x; % trial 차원 없으면 그대로
    end

    % --- plot (채널 전체) ---
    figure(1); clf
    plot(x_mean);  % time 축 자동, channel별 여러 라인
    xlabel("time");
    ylabel("signal (trial-mean)");
    title(sprintf("[%d/%d] %s | session %d | %s", i, numel(pairs), folder, sess, string(getfield(dir(fpath),'name'))), ...
          "Interpreter","none");
    grid on

    % --- control ---
    % Enter: 다음 sub
    % 1~5: session 선택
    % p: 이전 sub
    % q: 종료
    cmd = input("Enter=next sub | p=prev sub | 1~5=session | q=quit : ", "s");

    if isempty(cmd)
        i = i + 1;
        sess = 1;  % 다음 sub로 넘어갈 때 session은 1로 리셋(원하면 유지로 바꿔도 됨)
        if i > numel(pairs)
            break;
        end
        continue;
    end

    if strcmpi(cmd, "q")
        break;
    elseif strcmpi(cmd, "p")
        i = max(1, i-1);
        continue;
    elseif any(cmd == ["1","2","3","4","5"])
        sess = str2double(cmd);
        continue;
    else
        % 아무 키나: 그대로 유지
        continue;
    end
end