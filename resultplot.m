load('model_IMUCODE_result.mat')

sub = 3;      % 1~3
fold = 2;     % 1~5

%내 데이터 구조 : (num_subject, num_session_2, max_fine_epochs) - 1:train,2:val

train_acc = squeeze(fine_acc(sub, fold, 1, :));
val_acc   = squeeze(fine_acc(sub, fold, 2, :));

idx = ~isnan(train_acc);

figure
plot(find(idx), train_acc(idx)); hold on
plot(find(idx), val_acc(idx));
xlabel('epoch')
ylabel('acc')
legend('train','val')
title(sprintf('fine-acc | sub=%d fold=%d', sub, fold))

train_loss = squeeze(fine_loss(sub, fold, 1, :));
val_loss   = squeeze(fine_loss(sub, fold, 2, :));
idx = ~isnan(train_loss);

figure
plot(find(idx), train_loss(idx)); hold on
plot(find(idx), val_loss(idx));
xlabel('epoch')
ylabel('loss')
legend('train','val')
title(sprintf('fine-loss | sub=%d fold=%d', sub, fold))