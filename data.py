import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    
    def __init__(self, args, mode='train'):
        super().__init__()
        self.args = args
        self.mode = mode
        self.lookback = args.lookback
        self.horizon = args.horizon

        try:
            processed_data = np.load(f'{args.data_dir}/{args.dynamics}/dataset/{mode}_{self.args.lookback}_{self.args.horizon}.npz')
            self.X = torch.from_numpy(processed_data['X']).float().to(self.args.device)
            self.Y = torch.from_numpy(processed_data['Y']).float().to(self.args.device)
            self.mean_y, self.std_y = processed_data['mean_y'], processed_data['std_y']
        except:
            self.process()

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
    
    def process(self):
        # origin data: (sequence_length, node_num, feature_dim)
        # return: X: (sequence_length-lookback-horizon+1, lookback, node_num, feature_dim), Y: (sequence_length-lookback-horizon+1, horizon, node_num, feature_dim)
        simulation = np.load(f'{self.args.data_dir}/{self.args.dynamics}/dynamics.npz')['X']
        
        lookback = self.args.lookback
        horizon = self.args.horizon
                
        # Sliding window
        idx = np.arange(0, simulation.shape[0]-lookback-horizon+1)
        X = np.stack([simulation[i:i+lookback] for i in idx], axis=0)
        Y = np.stack([simulation[i+lookback:i+lookback+horizon] for i in idx], axis=0)
        
        # Normalize
        self.mean_x, self.mean_y = X.mean(axis=(0, 1, 2), keepdims=True), Y.mean(axis=(0, 1, 2), keepdims=True)
        self.std_x, self.std_y = X.std(axis=(0, 1, 2), keepdims=True), Y.std(axis=(0, 1, 2), keepdims=True)
        X = (X - self.mean_x) / self.std_x
        Y = (Y - self.mean_y) / self.std_y
        
        # Split train and test
        train_ratio, val_ratio = self.args.train_ratio, self.args.val_ratio
        train_size, val_ratio = int(X.shape[0] * train_ratio), int(X.shape[0] * val_ratio)
        train_idx = np.random.choice(np.arange(X.shape[0]), train_size, replace=False)
        val_idx = np.random.choice(np.setdiff1d(np.arange(X.shape[0]), train_idx), val_ratio, replace=False)
        test_idx = np.setdiff1d(np.arange(X.shape[0]), np.concatenate([train_idx, val_idx]))
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        
        # Save
        os.makedirs(f'{self.args.data_dir}/{self.args.dynamics}/dataset', exist_ok=True)
        np.savez(f'{self.args.data_dir}/{self.args.dynamics}/dataset/train_{self.args.lookback}_{self.args.horizon}.npz', X=X_train, Y=Y_train, mean_y=self.mean_y, std_y=self.std_y)
        np.savez(f'{self.args.data_dir}/{self.args.dynamics}/dataset/val_{self.args.lookback}_{self.args.horizon}.npz', X=X_val, Y=Y_val, mean_y=self.mean_y, std_y=self.std_y)
        np.savez(f'{self.args.data_dir}/{self.args.dynamics}/dataset/test_{self.args.lookback}_{self.args.horizon}.npz', X=X_test, Y=Y_test, mean_y=self.mean_y, std_y=self.std_y)
        
        if self.mode=='train':
            self.X, self.Y = X_train, Y_train
        elif self.mode=='val':
            self.X, self.Y = X_val, Y_val
        elif self.mode=='test':
            self.X, self.Y = X_test, Y_test
        
        # Convert to torch tensor
        self.X = torch.from_numpy(self.X).float().to(self.args.device)
        self.Y = torch.from_numpy(self.Y).float().to(self.args.device)
        
    def getLoader(self):
        return DataLoader(self, batch_size=self.args.batch_size, shuffle=True, drop_last=True)