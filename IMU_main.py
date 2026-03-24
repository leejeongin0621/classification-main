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

from torch.utils.tensorboard import SummaryWriter #TensorBoard for visualization
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# =========================
# Settings
# =========================
subject_list = ['250805_KDY','250814_JCM','250820_LTG' ]
#left='250804_KTS','250805_SMC','250811_LPR','250811_JHS','250812_HHJ','250813_YMS','250814_CYJ','250819_CYK','250822_KTH', '250827_HJH'
#right='250731_LGE','250805_KDY','250806_LJI','250812_WDY','250814_JCM','250818_ICY','250819_PYH','250820_LTG','250822_JSH','250825_JDB'
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

max_fine_epochs = 400
min_fine_epochs = 200
model_list = [1] # 1: IMU_BiLSTM, 2: IMU_conso_processing_Transformer 3: IMU_Transformer #ST-GCN 모델 
#4: IMU_STGCN (edge importance weighting x) 5: IMU_STGCN (no edge importance weighting o) #6 : IMU_STGCN (layer 6)

# =========================
# Data Load (IMU only)
# =========================
IMU_data_patient, IMU_label_patient = LoadIMU_EPO_simple(load_path, subject_list)
# shape: (subject, session, class, 200, 30)

# =========================
# Train (IMU only)
# =========================
for model_num in model_list:

    seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed) #fixed hash seed 
    random.seed(seed)  #Python  
    np.random.seed(seed) #Numpy
    torch.manual_seed(seed) #CPU
    torch.cuda.manual_seed(seed) #GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    print(f"\n=========== IMU model_{model_num} ==============\n")

    # ===== TensorBoard writer (one per run) =====
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logdir = os.path.join(save_path, "runs", f"IMU_m{model_num}_{run_id}")
    writer = SummaryWriter(log_dir=tb_logdir)
    print("TensorBoard logdir:", tb_logdir)

    fine_loss = np.empty((num_subject, num_session, 2, max_fine_epochs)) #make empty array for fine_loss and fine_acc
    fine_acc  = np.empty((num_subject, num_session, 2, max_fine_epochs))
    fine_loss.fill(np.nan) #Fill the array with NaN for epochs not reached during training
    fine_acc.fill(np.nan)

    ground_truth = np.empty((num_subject, num_session, (num_session-1)*num_class))
    prediction_result = np.empty((num_subject, num_session, (num_session-1)*num_class, num_class))
    ground_truth.fill(np.nan)
    prediction_result.fill(np.nan)

    for s, s_name in enumerate(subject_list): #enumerate for subject index and name

        print(f"\n##### model : {model_num} / test subject : {s_name} #####")

        sessions = np.arange(num_session)

        for fold in sessions:
            print(f"- fold {fold} -")

            test_session  = np.array([fold])
            train_session = np.delete(sessions, fold)

            # -------- IMU train/test split --------
            X_train = IMU_data_patient[s, train_session].reshape(-1, num_time_IMU, num_channel_IMU)
            y_train = IMU_label_patient[s, train_session].reshape(-1)

            X_test = IMU_data_patient[s, test_session].reshape(-1, num_time_IMU, num_channel_IMU)
            y_test = IMU_label_patient[s, test_session].reshape(-1)

            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            )

            train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=50, shuffle=True)

            # -------- Load IMU model --------
            model = torch.load(os.path.join(model_path, f"Model_6.pt"),
                               map_location=device,
                               weights_only=False)  # PyTorch 2.6 대응

            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0005) #3e-4,4e-4 not good
            #stepLR scheduler
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                             step_size=50,   # 50 epoch마다
            #                                             gamma=0.85       # lr 절반
            #                                             )
            
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_fine_epochs) -transformer-

            train_losses, val_losses = [], []
            train_accs, val_accs = [], []

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
                    outputs = model(X_IMU)   # ✅ IMU only
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    #scheduler.step() 
                    total_loss += loss.item() * X_IMU.size(0)
                    _, pred = outputs.max(1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)

                train_losses.append(total_loss / total)
                train_accs.append(correct / total)

                # ---- Validation ----
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0

                with torch.no_grad():
                    for X_IMU, y in val_loader:
                        X_IMU, y = X_IMU.to(device), y.to(device)
                        outputs = model(X_IMU)
                        loss = criterion(outputs, y)

                        val_loss += loss.item() * X_IMU.size(0)
                        _, pred = outputs.max(1)
                        val_correct += pred.eq(y).sum().item()
                        val_total += y.size(0)

                val_losses.append(val_loss / val_total)
                val_accs.append(val_correct / val_total)
                #current_lr = optimizer.param_groups[0]["lr"]

                global_step = (s * num_session + fold) * max_fine_epochs + epoch

                # scalars
                writer.add_scalar(f"IMU/{s_name}/fold{fold}/train_loss", train_losses[-1], global_step)
                writer.add_scalar(f"IMU/{s_name}/fold{fold}/train_acc",  train_accs[-1],  global_step)
                writer.add_scalar(f"IMU/{s_name}/fold{fold}/val_loss",   val_losses[-1],   global_step)
                writer.add_scalar(f"IMU/{s_name}/fold{fold}/val_acc",    val_accs[-1],    global_step)   
                #writer.add_scalar(f"IMU/{s_name}/fold{fold}/lr", current_lr, global_step)            

                if epoch >= min_fine_epochs - 1 and train_accs[-1] >= 0.99:
                    print(f"train early finished, epoch : {epoch}")
                    break

            # ---- Save curves ----
            L = len(train_accs)
            fine_acc[s, fold, 0, :L] = np.array(train_accs)
            fine_acc[s, fold, 1, :L] = np.array(val_accs)
            fine_loss[s, fold, 0, :L] = np.array(train_losses)
            fine_loss[s, fold, 1, :L] = np.array(val_losses)

            print(f"acc : train - {train_accs[-1]} / val - {val_accs[-1]} ")
            
            # ---- Save edge importance heatmap  ----
            # if model_num == 5 and hasattr(model, 'edge_importance'):
            import matplotlib.pyplot as plt
            import seaborn as sns

            for i, imp in enumerate(model.edge_importance):
                imp_np = imp.detach().cpu().numpy().sum(axis=0)  # (10, 10)
                
                plt.figure()
                sns.heatmap(imp_np, annot=True, fmt='.2f',
                            xticklabels=[f'N{j}' for j in range(10)],
                            yticklabels=[f'N{j}' for j in range(10)])
                plt.title(f"{s_name} fold {fold} layer {i}")
                plt.savefig(os.path.join(save_path, f"edge_{s_name}_fold{fold}_layer{i}.png")) 
                plt.close()

            # ---- Save model ----
            torch.save(
                model.state_dict(),
                os.path.join(model_path,
                             f"IMU_Model_1_finetuned_sub_{s_name}_fold_{fold}.pt")
            )

            # ---- Save prediction ----
            ground_truth[s, fold, :len(y_test)] = y_test

            with torch.no_grad():
                inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
                outputs = model(inputs)
                _, pred = outputs.max(1)

                prediction_result[s, fold, :len(pred), :] = (
                    nn.functional.one_hot(pred, num_classes=num_class)
                    .cpu().numpy()
                )

            del model
            torch.cuda.empty_cache() #empty all GPU cache for next fold
        

    # =========================
    # 5-Fold Summary
    # =========================
    valid_lens = np.sum(~np.isnan(fine_acc), axis=-1)
    last_indices = valid_lens - 1
    result = np.take_along_axis(fine_acc,
                                last_indices[..., np.newaxis],
                                axis=-1).squeeze(-1) #extract final accuracy of each fold (last trained epoch)

    result_mean = np.mean(result, axis=1)
    result_std  = np.std(result, axis=1)

    total_mean = np.mean(result_mean, axis=0)
    total_std  = np.std(result_mean, axis=0)

    print(f"\n======= IMU model_1 Average Acc =========")
    for s_idx, sname in enumerate(subject_list):
        print(f"{sname}: {result_mean[s_idx,1]} (±{result_std[s_idx,1]})")

    print(f"\nTotal Average Acc : {total_mean[1]} (±{total_std[1]})")

    # =========================
    # Save .mat
    # =========================
    io.savemat(
        os.path.join(save_path, f"model_1_result.mat"),
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