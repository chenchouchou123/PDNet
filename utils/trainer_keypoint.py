"""Training process
"""  
import os
import torch
import wandb
import torch.nn as nn
from torch.autograd import Variable
import tqdm
from utils.result_utils import *
import time
from datetime import datetime

from omegaconf import OmegaConf


class Trainer:

    def __init__(self, model, data_train, data_valid, data_test, data_test_video, args, device):
        self.model = model
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.data_test_video = data_test_video
        self.args = args
        self.device = device

    def train(self):
        self.model = self.model.to(self.device)
        loss_fn = nn.MSELoss().to(self.device)
        loss_fn_leg = nn.MSELoss().to(self.device)
        loss_fn_hand = nn.MSELoss().to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.train.learning_rate,
            weight_decay=self.args.train.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=1, eta_min=0.01*self.args.train.learning_rate
        )


        root_dir = getattr(self.args.train, "save_dir", "./checkpoints")
        os.makedirs(root_dir, exist_ok=True)
        run_stamp = datetime.now().strftime("%Y%m%d-%H%M")
        run_dir = os.path.join(root_dir, run_stamp)
        os.makedirs(run_dir, exist_ok=True)

        best_metric = float("inf")
        best_epoch = -1

        def _model_to_save(m):
            return m.module if hasattr(m, "module") else m

        def _save_ckpt(path, epoch_idx, test_mpjpe_val):
            ckpt = {
                "epoch": epoch_idx,
                "model_state": _model_to_save(self.model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
                "args": self.args,
                "metric": {"MPJPE": float(test_mpjpe_val)}
            }
            torch.save(ckpt, path)

        best_path = os.path.join(run_dir, "best.pt")
        last_path = os.path.join(run_dir, "last.pt")

        Epoch_num = self.args.train.epoch
        step = 0

        for epoch in range(Epoch_num):
            self.model.train()
            progress_bar = tqdm.tqdm(self.data_train)

            for iter, (x_batch, x_R_batch, y_batch) in enumerate(progress_bar):
                x_batch = Variable(x_batch.float().to(self.device))
                x_R_batch = Variable(x_R_batch.float().to(self.device))
                y_batch = Variable(y_batch.float().to(self.device))

                y_pred = self.model(x_batch, x_R_batch)

                loss_coord = loss_fn(y_pred.to(dtype=torch.float32), y_batch.to(dtype=torch.float32))
                loss_leg = loss_fn_leg(y_pred[:, :, (2,3,5,6), :], y_batch[:, :, (2,3,5,6), :])
                loss_hand = loss_fn_hand(y_pred[:, :, (12,13,15,16), :], y_batch[:, :, (12,13,15,16), :])
                loss_motion_leg = motion_cal(y_pred[:, :, (2,3,5,6), :], y_batch[:, :, (2,3,5,6), :], intervals=[2,4,6,8])
                loss_motion_hand = motion_cal(y_pred[:, :, (12,13,15,16), :], y_batch[:, :, (12,13,15,16), :], intervals=[2,4,6,8])

                loss = (
                    loss_coord
                    + (loss_leg + loss_hand) * self.args.train.alpha_limb
                    + (loss_motion_leg + loss_motion_hand) * self.args.train.alpha_limb_motion
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
            progress_bar.set_description(
                    'Step: {}. Epoch: {}/{}. Total: {:.3f}. Coord: {:.3f}. Motion_Leg: {:.3f}. Motion_Hand: {:.3f}'.format(
                        step, epoch+1, Epoch_num, loss.item(), loss_coord.item(),
                        loss_motion_leg.item()*0.05, loss_motion_hand.item()*0.05
                    )
                )


            test_loss, des_test = test_keypoint(self.data_test, self.device, self.model, output_temporal=True)
            cur_mpjpe = test_loss['MPJPE'].mean()
            cur_mpjpe_val = float(cur_mpjpe.item() if hasattr(cur_mpjpe, "item") else cur_mpjpe)

            print('test_MPJPE: {:.3f}. test_PCC: {:.3f}. test_PCK: {:.3f}%'.format(
                cur_mpjpe_val, test_loss['PCC'][:,1:].mean(), test_loss['PCK'].mean()*100))

            if self.args.wandb.use_wandb:
                wandb.log({
                    'lr': lr_scheduler.optimizer.param_groups[0]['lr'],
                    'test_MPJPE': cur_mpjpe_val,
                    'test_MPJPE_leg': test_loss['MPJPE'][:,:,:,(2,3,5,6)].mean(),
                    'test_MPJPE_hand': test_loss['MPJPE'][:,:,:,(12,13,15,16)].mean(),
                    'test_PCC': test_loss['PCC'][:,:,:,1:].mean(),
                    'test_PCC_leg': test_loss['PCC'][:,:,:,(2,3,5,6)].mean(),
                    'test_PCC_hand': test_loss['PCC'][:,:,:,(12,13,15,16)].mean(),
                    'test_PCK': test_loss['PCK'].mean()*100,
                    'test_PCK_leg': test_loss['PCK'][:,:,:,(2,3,5,6)].mean()*100,
                    'test_PCK_hand': test_loss['PCK'][:,:,:,(12,13,15,16)].mean()*100,
                })


            _save_ckpt(last_path, epoch_idx=epoch+1, test_mpjpe_val=cur_mpjpe_val)

            if cur_mpjpe_val < best_metric:
                best_metric = cur_mpjpe_val
                best_epoch = epoch + 1
                _save_ckpt(best_path, epoch_idx=best_epoch, test_mpjpe_val=best_metric)
                print(f"[CKPT] New best at epoch {best_epoch}: MPJPE={best_metric:.4f} -> {best_path}")


            lr_scheduler.step()


        meta_path = os.path.join(run_dir, "best_epoch.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(
                f"Best Epoch: {best_epoch}\n"
                f"Best MPJPE: {best_metric:.4f}; PCK:{test_loss['PCK'].mean()*100}\n"
                f"Checkpoint Path: {best_path}\n"
            )
