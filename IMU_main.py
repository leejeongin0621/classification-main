import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils.Dataloader import * # * means load all dataloader.py functions
import os
from scipy import io
import random
from tqdm import tqdm
from correlation import make_sample_corr, make_word_avg_corr, make_word_corr_dict

from torch.utils.tensorboard import SummaryWriter #TensorBoard for visualization
from datetime import datetime
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# =========================
# Settings
# =========================
subject_list = ['250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM','250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB']
#left='250804_KTS','250805_SMC','250811_LPR','250811_JHS','250812_HHJ','250813_YMS','250814_CYJ','250819_CYK','250822_KTH', '250827_HJH'
#right='250805_KDY','250731_LGE','250806_LJI','250812_WDY','250814_JCM','250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB'
num_subject = len(subject_list)
num_session = 5

fs = 50
num_time_IMU = fs * 4
num_class = 100
num_channel_IMU = 30

this_path = os.getcwd() # Get current working directory
# path_mom = this_path.split("1.research")[0]
# path_proj = os.path.join(path_mom, "1.research", '2025_SSI', 'I2T')
# path_sub_proj = os.path.join(path_proj, 'classification')

path_sub_proj = this_path

load_path  = os.path.join(path_sub_proj, 'data_10ch')
save_path  = os.path.join(path_sub_proj, 'results')
model_path = os.path.join(path_sub_proj, 'models')

os.makedirs(save_path, exist_ok=True) #Create directory if it does not exist
os.makedirs(model_path, exist_ok=True)

max_fine_epochs = 250
min_fine_epochs = 200
model_list = [1] # 1: IMU_BiLSTM, 2: IMU_conso_processing_Transformer 3: IMU_Transformer #ST-GCN 모델 
#4: IMU_STGCN (edge importance weighting x) 5: IMU_STGCN (no edge importance weighting o) #6 : IMU_STGCN (layer 6) #7 : IMU_STGCN (layer 8) 
#8 : IMU_STGCN (layer 8),kernel 9

# =========================
# Data Load (IMU only)
# =========================
IMU_data_patient, IMU_label_patient = LoadIMU_EPO_simple(load_path, subject_list)
# shape: (subject, session, class, 200, 30)

# =========================
# Train (IMU only)
# =========================
for model_num in model_list:

    # seed = 2023
    # os.environ['PYTHONHASHSEED'] = str(seed) #fixed hash seed 
    # random.seed(seed)  #Python  
    # np.random.seed(seed) #Numpy
    # torch.manual_seed(seed) #CPU
    # torch.cuda.manual_seed(seed) #GPU
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False 
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    
    subject_id_map = {
        '250805_KDY': 1,
        '250731_LGE': 2,
        '250806_LJI': 3,
        '250812_WDY': 4,
        '250814_JCM': 5,
        '250818_ICY': 6,
        '250819_PYH': 7,
        '250820_LTG': 8,
        '250822_JSH': 9,
        '250825_JDB': 10,
    }

    base_seed = 2023

    print(f"\n=========== IMU model_{model_num} ==============\n")

    # ===== TensorBoard writer (one per run) =====
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logdir = os.path.join(save_path, "runboard", f"IMU_m{model_num}_{run_id}")
    writer = SummaryWriter(log_dir=tb_logdir)
    print("TensorBoard logdir:", tb_logdir)

    fine_loss = np.empty((num_subject, num_session, 2, max_fine_epochs)) #make empty array for fine_loss and fine_acc
    fine_acc  = np.empty((num_subject, num_session, 2, max_fine_epochs))
    fine_loss.fill(np.nan) #Fill the array with NaN for epochs not reached during training
    fine_acc.fill(np.nan)

    best_acc_result = np.empty((num_subject, num_session))
    best_loss_result = np.empty((num_subject, num_session))
    best_acc_result.fill(np.nan)
    best_loss_result.fill(np.nan)

    ground_truth = np.empty((num_subject, num_session, (num_session-1)*num_class))
    prediction_result = np.empty((num_subject, num_session, (num_session-1)*num_class, num_class))
    ground_truth.fill(np.nan)
    prediction_result.fill(np.nan)

    for s, s_name in enumerate(subject_list): #enumerate for subject index and name

        print(f"\n##### model : {model_num} / test subject : {s_name} #####")

        sessions = np.arange(num_session)

        for fold in sessions:

            seed = base_seed + subject_id_map[s_name] * 100 + int(fold)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = False
            print(f"- fold {fold} -")

            test_session  = np.array([fold])
            train_session = np.delete(sessions, fold)

            # -------- IMU train/test split --------
            X_train = IMU_data_patient[s, train_session].reshape(-1, num_time_IMU, num_channel_IMU)
            y_train = IMU_label_patient[s, train_session].reshape(-1)

            X_test = IMU_data_patient[s, test_session].reshape(-1, num_time_IMU, num_channel_IMU)
            y_test = IMU_label_patient[s, test_session].reshape(-1)

            word_corr_dict = make_word_corr_dict(X_train, y_train)

            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            )

            train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

            # -------- Load IMU model --------
            model = torch.load(os.path.join(model_path, f"Model_37.pt"),
                               map_location=device,weights_only=False)  # PyTorch 2.6 대응

            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001) #3e-4,4e-4 not good 원래 0.0005 ,0.0005, weight_decay=1e-4
            #stepLR scheduler
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                             step_size=50,   # 50 epoch마다
            #                                             gamma=0.85       # lr 절반
            #                                             )
            
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_fine_epochs) -transformer-

            train_losses, val_losses = [], []
            train_accs, val_accs = [], []

            best_model_path = os.path.join(model_path, f"best_model_{s_name}_fold_{fold}.pth")
            best_val_acc = -1
            best_epoch = -1
            counter = 0
            best_val_loss= 0
            patience=15

            # -------- Training --------
            for epoch in tqdm(
                range(max_fine_epochs),
                desc=f"Train m{model_num} {s_name} fold {fold}",
                leave=False
            ): 

                
                model.train()
                total_loss, correct, total = 0.0, 0, 0

                for X_IMU, y in train_loader:
                    X_IMU, y = X_IMU.to(device), y.to(device)

                    optimizer.zero_grad() #backpropagation
                    
                    for i in range(X_IMU.size(0)):
                        X_sample = X_IMU[i:i+1]  # (200, 30)
                        y_sample = y[i:i+1]      # (1,)

                        word_label = int(y_sample.item()) #단어마다 adjacency 만들기 
                        A_corr_np = word_corr_dict[word_label]
                        A_corr = torch.tensor(A_corr_np, dtype=torch.float32, device=device)

                    outputs = model(X_IMU)   # ✅ IMU only
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step() 
                    total_loss += loss.item() * X_IMU.size(0)
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)

                train_losses.append(total_loss / total)
                train_accs.append(correct / total)

                # ---- Validation ----
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0

                model.eval()

                node_importance_total = torch.zeros(10, device=device)

                for X_IMU, y in val_loader:
                    X_IMU, y = X_IMU.to(device), y.to(device)

                    X_IMU.requires_grad_(True)

                    # 단어별 adjacency 만들기-test 셋으로만 
                    x_np = X_IMU[0].detach().cpu().numpy()   # (200, 30)
                    A_corr_np = make_sample_corr(x_np)
                    A_corr = torch.tensor(A_corr_np, dtype=torch.float32, device=device)

                    outputs= model(X_IMU)   # ✅ IMU only
                    loss = criterion(outputs, y)
                    val_loss += loss.item() * X_IMU.size(0)
                    _, pred = outputs.max(1)
                    val_correct += pred.eq(y).sum().item()
                    val_total += y.size(0)

                    target = outputs[torch.arange(len(pred), device=device), pred].mean()

                    model.zero_grad()
                    target.backward()

                    grad = X_IMU.grad  # (B, T, 30)
                    grad = grad.view(grad.size(0), grad.size(1), 10, 3)  # (B, T, V, C)

                    importance = grad.abs().mean(dim=(0, 1, 3))  # (10,)
                    node_importance_total += importance.detach()

                    X_IMU.grad = None

                node_importance = node_importance_total / len(val_loader)

                # with torch.no_grad():
                #     for X_IMU, y in val_loader:
                #         X_IMU, y = X_IMU.to(device), y.to(device)
                #         outputs = model(X_IMU)
                #         loss = criterion(outputs, y)

                #         val_loss += loss.item() * X_IMU.size(0)
                #         _, pred = outputs.max(1)
                #         val_correct += pred.eq(y).sum().item()
                #         val_total += y.size(0)

                val_losses.append(val_loss / val_total)
                val_accs.append(val_correct / val_total)
                #current_lr = optimizer.param_groups[0]["lr"]

                global_step = (s * num_session + fold) * max_fine_epochs + epoch

                # scalars
                writer.add_scalar(f"IMU/{s_name}/fold{fold}/train_loss", train_losses[-1], global_step)
                writer.add_scalar(f"IMU/{s_name}/fold{fold}/train_acc",  train_accs[-1],  global_step)
                writer.add_scalar(f"IMU/{s_name}/fold{fold}/val_loss",   val_losses[-1],   global_step)
                writer.add_scalar(f"IMU/{s_name}/fold{fold}/val_acc",    val_accs[-1],    global_step)   

                # if hasattr(model, 'edge_importance'):
                #     imp_last = model.edge_importance[-1].detach().cpu().numpy()  # (K, V, V) or (V, V)

                #     # branch 평균 인데 난 없긴 함.. 
                #     if imp_last.ndim == 3:
                #         imp_last = imp_last.mean(axis=0)  # (V, V)

                #     V = imp_last.shape[0]

                #     for i in range(V):
                #         for j in range(i + 1, V):   # 중복 제거 (undirected)
                #             writer.add_scalar(
                #                 f"EdgeCurve/{s_name}/fold{fold}/edge_{i}_{j}",
                #                 float(imp_last[i, j]),
                #                 epoch
                #             )
                                


                current_val_acc = val_accs[-1]
                current_val_loss = val_losses[-1]

                improved = current_val_acc > best_val_acc

                if improved:
                    best_val_acc = current_val_acc
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_path)

                if epoch >= min_fine_epochs - 1:
                    if improved:
                        counter = 0
                    else:
                        counter += 1

                    if counter >= patience:
                        print(f"early stop at epoch {epoch}, best epoch was {best_epoch}")
                        break 

            # ---- Save curves ----
            L = len(train_accs)
            fine_acc[s, fold, 0, :L] = np.array(train_accs)
            fine_acc[s, fold, 1, :L] = np.array(val_accs)
            fine_loss[s, fold, 0, :L] = np.array(train_losses)
            fine_loss[s, fold, 1, :L] = np.array(val_losses)

            best_acc_result[s, fold] = best_val_acc
            best_loss_result[s, fold] = best_val_loss

            print(f"acc : train - {train_accs[-1]} / val - {best_val_acc} ")
            
            # ---- Save edge importance heatmap  ----
            # if model_num == 5 and hasattr(model, 'edge_importance'):
            import matplotlib.pyplot as plt
            import seaborn as sns

            # A_np = model.A.detach().cpu().numpy()                 # (K, 10, 10)
            # imp_last = model.edge_importance[-1].detach().cpu().numpy()   # (K, 10, 10)
        

    # =========================
    # 5-Fold Summary
    # =========================
    # valid_lens = np.sum(~np.isnan(fine_acc), axis=-1)
    # last_indices = valid_lens - 1
    # result = np.take_along_axis(fine_acc,
    #                             last_indices[..., np.newaxis],
    #                             axis=-1).squeeze(-1) #extract final accuracy of each fold (last trained epoch)

    result_mean = np.mean(best_acc_result, axis=1)
    result_std  = np.std(best_acc_result, axis=1)

    total_mean = np.mean(result_mean, axis=0)
    total_std  = np.std(result_mean, axis=0)

    print(f"\n======= IMU model_14_average Acc =========")
    for s_idx, sname in enumerate(subject_list):
        print(f"{sname}: {result_mean[s_idx]} (±{result_std[s_idx]})")

    print(f"\nTotal Average Acc : {total_mean} (±{total_std})")

    # =========================
    # Save .mat
    # =========================
    io.savemat(
        os.path.join(save_path, f"model_14_result.mat"),
        {
            'fine_loss': fine_loss,
            'fine_acc': fine_acc,
            'ground_truth': ground_truth,
            'prediction': prediction_result,
            'subject_list': subject_list
        }
    )

    print("\nresult saved!\n")

writer.flush()
writer.close()
