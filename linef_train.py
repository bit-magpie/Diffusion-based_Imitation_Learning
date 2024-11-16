from argparse import ArgumentParser
import os
from collections import OrderedDict
from itertools import product

import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm.contrib import tenumerate
from torchvision import transforms

from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

from diff_model import (
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)

DATASET_PATH = "datasets/line_follow"
SAVE_DATA_DIR = "diff_weights"  # for models/data

n_epoch = 100
lrate = 1e-4
device = "cuda"
n_hidden = 512
batch_size = 64
n_T = 50
net_type = "transformer"


class LineFollowDataset(Dataset):
    def __init__(
        self, DATASET_PATH, transform=None, train_or_test="train", train_prop=0.90
    ):
        self.DATASET_PATH = DATASET_PATH
        # just load it all into RAM
        self.images = np.load(os.path.join(DATASET_PATH, "imgs.npy"), allow_pickle=True)
        self.actions = np.load(os.path.join(DATASET_PATH, "pts.npy"), allow_pickle=True)
        self.transform = transform
        n_train = int(self.actions.shape[0] * train_prop)
        if train_or_test == "train":
            self.actions = self.actions[:n_train]
        elif train_or_test == "test":
            self.actions = self.actions[n_train:]
        else:
            raise NotImplementedError

        # normalise actions and images to range [0,1]
        # self.actions = self.actions / 500.0
        # self.images = self.images / 255.0

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, index):
        action = self.actions[index] 
        image = self.images[int(action[0])]
        action = action[1:]
        
        if self.transform:
            image = self.transform(image)
        return (image, action)

def get_model(x_shape, y_dim):
    nn_model = Model_cnn_mlp(
        x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type
    ).to(device)
     
    return Model_Cond_Diffusion(
        nn_model,
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        x_dim=x_shape,
        y_dim=y_dim,
        drop_prob=0.1,
        guide_w=0.0,
    )

def train_claw(n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type):
    # Unpack experiment settings
    # exp_name = experiment["exp_name"]
    # model_type = experiment["model_type"]
    # drop_prob = experiment["drop_prob"]

    # get datasets set up
    tf = transforms.Compose([])
    torch_data_train = LineFollowDataset(
        DATASET_PATH, transform=tf, train_or_test="train", train_prop=0.90
    )
    dataload_train = DataLoader(
        torch_data_train, batch_size=batch_size, shuffle=True, num_workers=0
    )

    x_shape = torch_data_train.images[0].shape
    y_dim = torch_data_train.actions.shape[1] - 1

    # create model
    model = get_model(x_shape, y_dim)

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lrate)

    for ep in tqdm(range(n_epoch), desc="Epoch"):
        results_ep = [ep]
        model.train()

        # lrate decay
        optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

        # train loop
        pbar = tqdm(dataload_train)
        loss_ep, n_batch = 0, 0
        for x_batch, y_batch in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(device)
            y_batch = y_batch.type(torch.FloatTensor).to(device)
            # print(x_batch.shape, y_batch.shape)
            loss = model.loss_on_batch(x_batch, y_batch)
            optim.zero_grad()
            loss.backward()
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
            optim.step()
        results_ep.append(loss_ep / n_batch)

    torch.save(model.state_dict(), os.path.join(SAVE_DATA_DIR, "trained_model.pt"))

def predict(model, x_eval):
    model.eval()
    use_kde = False
    
    with torch.set_grad_enabled(False):
        # if extra_diffusion_step == 0:
        y_pred_ = (
            model.sample(x_eval, extract_embedding=True)
            .detach()
            .cpu()
            .numpy()
        )

        if use_kde:
            # kde
            torch_obs_many = x_eval
            action_pred_many = model.sample(torch_obs_many).cpu().numpy()
            # fit kde to the sampled actions
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(action_pred_many)
            # choose the max likelihood one
            log_density = kde.score_samples(action_pred_many)
            idx = np.argmax(log_density)
            y_pred_ = action_pred_many[idx][None, :]
        else:
            y_pred_ = model.sample_extra(x_eval, extra_steps=16).detach().cpu().numpy()
    
    return y_pred_
           


def eval_model():
    tf = transforms.Compose([])
    torch_data_train = LineFollowDataset(
        DATASET_PATH, transform=tf, train_or_test="train", train_prop=0.90
    )
    
    x_shape = torch_data_train.image_all.shape[1:]
    y_dim = torch_data_train.action_all.shape[1]
    
    model = get_model(x_shape, y_dim)
    
    model.eval()
    use_kde = False
    idxs = [14, 2, 0, 9, 5, 35, 16]
    
    idxs_data = [[] for _ in range(len(idxs))]
    # for extra_diffusion_step, use_kde in product(extra_diffusion_steps, use_kdes):
        # if extra_diffusion_step != 0 and use_kde:
        #     continue
    for i, idx in tenumerate(idxs):
        x_eval = (
            torch.Tensor(torch_data_train.image_all[idx])
            .type(torch.FloatTensor)
            .to(device)
        )
        
        x_eval_large = torch_data_train.image_all_large[idx]
        obj_mask_eval = torch_data_train.label_all[idx]
        if i == 0:
            obj_mask_eval_marginal = np.zeros_like(obj_mask_eval)
        obj_mask_eval_marginal += obj_mask_eval
        
        for j in range(6 if not use_kde else 300):
            x_eval_ = x_eval.repeat(50, 1, 1, 1)
            y_pred_ = predict(model, x_eval_)
            
            if j == 0:
                y_pred = y_pred_
            else:
                y_pred = np.concatenate([y_pred, y_pred_])
        x_eval = x_eval.detach().cpu().numpy()

        idxs_data[i] = {
            "idx": idx,
            "x_eval_large": x_eval_large,
            "obj_mask_eval": obj_mask_eval,
            "y_pred": y_pred,
        }

        # Save data as a pickle
        # true_exp_name = exp_name
        # if extra_diffusion_step != 0:
        #     true_exp_name = f"{exp_name}_extra-diffusion_{extra_diffusion_step}"
        # if use_kde:
        #     true_exp_name = f"{exp_name}_kde"
        # if guide_weight is not None:
        #     true_exp_name = f"{exp_name}_guide-weight_{guide_weight}"
        with open(os.path.join(SAVE_DATA_DIR, f"diff_only.pkl"), "wb") as f:
            pickle.dump(idxs_data, f)


if __name__ == "__main__":
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    
    print("Training the model")
    train_claw(n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type)
    # print("Evaluating the model")
    # eval_model()
