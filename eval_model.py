import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

from diff_model import (
    Model_cnn_mlp,
    Model_Cond_Diffusion,
)

from sklearn.neighbors import KernelDensity

device = "cuda"

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
    
    n_epoch = 100
    lrate = 1e-4    
    n_hidden = 512
    batch_size = 32
    n_T = 50
    net_type = "transformer"
    
    nn_model = Model_cnn_mlp(
        x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type
    ).to(device)
    model = Model_Cond_Diffusion(
        nn_model,
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        x_dim=x_shape,
        y_dim=y_dim,
        drop_prob=0.1,
        guide_w=0.0,
    )
    
    return model

def predict(model, x_eval):
    model.eval()
    use_kde = True
    
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

def eval():
    DATASET_PATH = "datasets/line_follow/"
    save_dir = "results/"
    os.makedirs(save_dir, exist_ok=True)
    
    tf = transforms.Compose([])

    torch_data_train = LineFollowDataset(
            DATASET_PATH, transform=tf, train_or_test="train", train_prop=0.90
    )
    
    x_shape = torch_data_train.images[0].shape
    y_dim = torch_data_train.actions.shape[1] - 1
    
    model = get_model(x_shape, y_dim)
    
    checkpoint = torch.load("diff_weights/trained_model.pt", weights_only=True)
    model.load_state_dict(checkpoint)
    
    for i in tqdm(range(24)):
        x_eval = (
            torch.Tensor(torch_data_train.images[i])
            .type(torch.FloatTensor)
            .to(device)
        )
        
        x_eval_ = x_eval.repeat(100, 1, 1, 1)
        pred = predict(model, x_eval_)
        
        pred = pred.to_device('cpu') * 32
        x_eval = x_eval.detach().cpu()
        
        plt.figure()
        plt.imshow(x_eval)
        plt.scatter(pred[:, 0], pred[:, 1])
        plt.savefig(os.path.join(save_dir, f'plot{i}.png'), bbox_inches = 'tight')
        
if __name__ == "__main__":
    eval()