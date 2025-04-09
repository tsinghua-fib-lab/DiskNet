import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import mercator
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import torchdiffeq as ode

from utils import drawTraj, draw_embedding


def normalized_laplacian(A: torch.Tensor):
    """Symmetrically Normalized Laplacian: I - D^-1/2 * ( A ) * D^-1/2"""
    out_degree = torch.sum(A, dim=1)
    int_degree = torch.sum(A, dim=0)
    
    out_degree_sqrt_inv = torch.pow(out_degree, -0.5)
    int_degree_sqrt_inv = torch.pow(int_degree, -0.5)
    mx_operator = torch.eye(A.shape[0], device=A.device) - torch.diag(out_degree_sqrt_inv) @ A @ torch.diag(int_degree_sqrt_inv)
    
    return mx_operator


class HyperbolicEmbedding:
    def __init__(self, args: dict):
        self.args = args
    
    def fit_transform(self):
        if not os.path.exists(f'{self.args.log_dir}/HE/he.inf_coord'):
            os.makedirs(f'{self.args.log_dir}/HE', exist_ok=True)
            mercator.embed(
                edgelist_filename=f'{self.args.data_dir}/graph.txt', 
                quiet_mode=self.args.quiet_mode, 
                fast_mode=self.args.fast_mode, 
                output_name=f'{self.args.log_dir}/HE/he', 
                validation_mode=self.args.validation_mode, 
                post_kappa=self.args.post_kappa
            )
         
        if self.args.refine:
             mercator.embed(
                edgelist_filename=f'{self.args.data_dir}/graph.txt', 
                quiet_mode=self.args.quiet_mode, 
                fast_mode=self.args.fast_mode, 
                output_name=f'{self.args.log_dir}/HE/he', 
                validation_mode=self.args.validation_mode, 
                post_kappa=self.args.post_kappa,
                inf_coord=f'{self.args.log_dir}/HE/he.inf_coord'
            )
                
        return self._parse_mercator_output()

    def _parse_mercator_output(self):
        with open(f'{self.args.log_dir}/HE/he.inf_coord', 'r') as f:
            lines = f.readlines()
        
        # parse node_num, dim, coords
        node_num = int(lines[7].split()[-1])
        beta = float(lines[8].split()[-1])
        mu = float(lines[9].split()[-1])
        
        kappa = np.zeros(node_num)
        angular = np.zeros(node_num)
        radius = np.zeros(node_num)
        for i in range(15, 15+node_num):
            kappa[i-15] = float(lines[i].split()[1])
            angular[i-15] = float(lines[i].split()[2])
            radius[i-15] = float(lines[i].split()[3])
        
        return kappa, angular, radius


def atanh(x, eps=1e-5):
    x = torch.clamp(x, max=1. - eps)
    return .5 * (torch.log(1 + x) - torch.log(1 - x))

class PoincareManifold:
    
    @staticmethod
    def poincare_grad(euclidean_grad, x, c=-1, eps=1e-5):
        """
        Compute the gradient of the Poincare distance with respect to x.
        """
        sqnormx = torch.sum(x * x, dim=-1, keepdim=True)
        result = ((1 + c*sqnormx) / 2) ** 2 * euclidean_grad
        return result
    
    @staticmethod
    def log_map_zero(x, c=-1, eps=1e-5):
        """
        Log map from Poincare ball space to tangent space of zero.
        Ref: 
        1. https://github.com/cll27/pvae/tree/7abbb4604a1acec2332b1b4dfe21267834b505cc
        2. https://github.com/facebookresearch/hgnn/blob/master/manifold/PoincareManifold.py
        """
        norm_diff = torch.norm(x, 2, dim=1, keepdim=True)
        atanh_x = atanh(np.sqrt(np.abs(c)) * norm_diff)
        lam_zero = 2. # lambda = 2 / (1 + ||zero||) = 2
        return 2. / (np.sqrt(np.abs(c)) * lam_zero) * atanh_x * (x + eps) / norm_diff


class GNN(nn.Module):
    def __init__(self, feature_dim, ode_hid_dim):
        super(GNN, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(feature_dim, ode_hid_dim, bias=True), 
            nn.ReLU(),
            nn.Linear(ode_hid_dim, ode_hid_dim, bias=True),
        )
        self.f2 = nn.Sequential(
            nn.Linear(ode_hid_dim, ode_hid_dim, bias=True), 
            nn.ReLU(),
            nn.Linear(ode_hid_dim, feature_dim, bias=True),
        )
        
        self.adj = None

    def forward(self, x):
        x = self.f1(x)
        x = self.adj @ x
        x = self.f2(x)
        return x
    
    
class BackboneODE(nn.Module):
    """dXdt = f(X) + g(X, A)"""
    def __init__(self, lookback, feature_dim, ode_hid_dim, method):
        super(BackboneODE, self).__init__()
        
        self.method = method
        self.feature_dim = feature_dim
        self.init_enc = nn.Sequential(
            nn.Linear(lookback, ode_hid_dim, bias=True), 
            nn.ReLU(),
            nn.Linear(ode_hid_dim, 1, bias=True)
        )
        self.f = nn.Sequential(
            nn.Linear(feature_dim, ode_hid_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ode_hid_dim, feature_dim, bias=True),
        )
        self.g = GNN(feature_dim, ode_hid_dim)
        
    def dxdt(self, t, x):
        x_self = self.f(x)
        x_neigh = self.g(x)
        dxdt = x_self + x_neigh
        return dxdt

    def forward(self, tspan, x, adj_w):
        # batch_size, lookback, node_num, feature_dim
        self.g.adj = adj_w
        
        x = x.permute(0, 2, 3, 1) # batch_size, node_num, feature_dim, lookback
        x = self.init_enc(x) # batch_size, node_num, feature_dim, 1
        x = x.squeeze(-1) # batch_size, node_num, feature_dim
        out = ode.odeint(self.dxdt, x, tspan, method=self.method) # horizon, batch_size, node_num, feature_dim
        out = out.permute(1, 0, 2, 3) # batch_size, horizon, node_num, feature_dim
        return out
    
    
class Refiner(nn.Module):
    def __init__(self, lookback, horizon, feature_dim, hid_dim):
        super(Refiner, self).__init__()
        
        self.feature_dim = feature_dim
        self.mlp_X = nn.Sequential(
            nn.Linear(lookback*feature_dim, hid_dim),
            nn.Tanh(),
        )
        self.mlp_Y = nn.Sequential(
            nn.Linear(horizon*feature_dim, hid_dim),
            nn.Tanh(),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, horizon*feature_dim),
        )
        
    def forward(self, X, Y):
        X = X.permute(0, 2, 1, 3) # batch_size, node_num, lookback, feature_dim
        X = X.reshape(X.shape[0], X.shape[1], -1) # batch_size, node_num, lookback*feature_dim
        Y = Y.permute(0, 2, 1, 3) # batch_size, node_num, lookback, feature_dim
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1) # batch_size, node_num, lookback*feature_dim
        
        X = self.mlp_X(X)
        Y = self.mlp_Y(Y)
        output = torch.cat([X, Y], dim=-1)
        refined_Y = self.mlp_out(output)
        
        refined_Y = refined_Y.reshape(refined_Y.shape[0], refined_Y.shape[1], -1, self.feature_dim) # batch_size, node_num, horizon, feature_dim
        refined_Y = refined_Y.permute(0, 2, 1, 3) # batch_size, horizon, node_num, feature_dim
        return refined_Y


class DiskNet(nn.Module):
    
    def __init__(self, args, adj):
        super(DiskNet, self).__init__()
        self.args = args
        self.model_args = args['DiskNet']
        
        self.adj = torch.from_numpy(adj).float().to(args.device)
        self.norm_lap = normalized_laplacian(self.adj)
        self.feature_dim = args[args.dynamics].dim
        
        # Identity Backbone
        self.repr_net1 = nn.Sequential(
            nn.Linear(self.model_args.n_dim, self.model_args.ag_hid_dim),
            nn.Tanh(),
            nn.Linear(self.model_args.ag_hid_dim, self.model_args.ag_hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.model_args.ag_hid_dim),
        )
        self.repr_net2 = nn.Sequential(
            nn.Linear(self.model_args.n_dim, self.model_args.ag_hid_dim),
            nn.Tanh(),
            nn.Linear(self.model_args.ag_hid_dim, self.model_args.ag_hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.model_args.ag_hid_dim),
        )
        self.softmax = nn.Softmax(dim=-1)
        
        # State aggregation
        self.agc_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.model_args.ag_hid_dim),
            nn.ReLU(),
            nn.Linear(self.model_args.ag_hid_dim, self.feature_dim),
        )
        self.tanh = nn.Tanh()
        
        # Backbone Dynamics
        self.BackboneODE = BackboneODE(args.lookback, self.feature_dim, self.model_args.ode_hid_dim, self.model_args.method)
        
        # K-means
        self.cluster_idx, self.cluster_centers = self._kmeans(adj, self.model_args.k, self.model_args.log)
        
        # Refine
        self.refiners = nn.ModuleList([Refiner(args.lookback, args.horizon, self.feature_dim, self.model_args.sr_hid_dim) for _ in range(self.model_args.k)])
        
        # Device
        self.to(args.device)
        
        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
        # Init hyperbolic embedding
        self.node_embedding, angular = self._init_poincare()
        self.supernode_embedding, backbone, assignment_matrix = self._init_super_node(angular)
        draw_embedding(self.adj, self.node_embedding, f'{self.args.log_dir}/HE/init_node_poincare.png')
        draw_embedding(backbone, self.supernode_embedding, f'{self.args.log_dir}/HE/init_supernode_poincare.png')
        
        if self.model_args.prior_init:
            # Pretrain for Identity Backbone
            self._pretrain_identity_backbone(assignment_matrix)
            draw_embedding(self.backbone, self.supernode_embedding, f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/pretrain_supernode_poincare.png')
    
    def _init_poincare(self):
        print('Initializing poincare embedding...')
        
        _, angular, radius = HyperbolicEmbedding(self.args).fit_transform()
        
        # Poincaré Disk to Euclidean
        x = np.tanh(radius / 2) * np.cos(angular)
        y = np.tanh(radius / 2) * np.sin(angular)
        poincare_embedding = torch.from_numpy(np.stack([x, y], axis=1)).float().to(self.args.device)
        
        print('Done.')
        return self._check_norm(poincare_embedding), angular
    
    def _init_super_node(self, angular):
        print('Initializing super node embedding...')
        num = int(self.model_args.ratio * self.args.node_num)
        
        # init super node embedding by angular
        idx = np.argsort(angular)
        assignment_matrix = torch.zeros(num, self.args.node_num).to(self.args.device)
        size = int(1/self.model_args.ratio)
        for i in range(num):
            assignment_matrix[i, idx[i*size:(i+1)*size]] = 1
        
        degree = self.adj.sum(axis=1, keepdims=True)
        super_node_embedding = assignment_matrix @ (self.node_embedding * degree) / (assignment_matrix @ degree).sum(dim=-1, keepdim=True)
        super_node_embedding = nn.Parameter(self._check_norm(super_node_embedding))
        backbone = assignment_matrix @ self.adj @ assignment_matrix.T
                
        print('Done.')
        return super_node_embedding, backbone, assignment_matrix
    
    def _pretrain_identity_backbone(self, prior_assignment_matrix):
        
        optimizer = torch.optim.Adam(
            [
                {'params': self.repr_net1.parameters(), 'lr': self.args.lr},
                {'params': self.repr_net2.parameters(), 'lr': self.args.lr},
                {'params': self.agc_mlp.parameters(), 'lr': self.args.lr},
            ], 
            lr=self.args.lr)
        loss_fn = nn.L1Loss()
        
        for epoch in range(self.model_args.pretrain_epoch):
            # 1. map to euclidean space from poincare space
            node_euclidean_embedding = PoincareManifold.log_map_zero(self.node_embedding)
            supernode_euclidean_embedding = PoincareManifold.log_map_zero(self.supernode_embedding)
            # 2. topology-aware representation
            node_repr = self.repr_net1(node_euclidean_embedding)
            supernode_repr = self.repr_net2(supernode_euclidean_embedding)
            # 3. assignment matrix
            assignment_prob = self.softmax(supernode_repr @ node_repr.T)
            assignment_matrix = assignment_prob
            # 4. loss
            loss = loss_fn(assignment_matrix, prior_assignment_matrix)
            # 5. update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'\rPretrain identity backbone[{epoch}]: {loss.item():.4f}', end='')
        
        print()
        del optimizer, loss_fn
    
    @property
    def assignment_matrix(self):
        # 1. map to euclidean space from poincare space
        node_euclidean_embedding = PoincareManifold.log_map_zero(self.node_embedding)
        supernode_euclidean_embedding = PoincareManifold.log_map_zero(self.supernode_embedding)
        # 2. topology-aware representation
        node_repr = self.repr_net1(node_euclidean_embedding)
        supernode_repr = self.repr_net2(supernode_euclidean_embedding)
        # 3. assignment matrix
        assignment_prob = self.softmax(supernode_repr @ node_repr.T)
        assignment_matrix = assignment_prob
        return assignment_matrix
    
    @property
    def backbone(self):
        assignment_matrix = self.assignment_matrix
        idx = torch.argmax(assignment_matrix, dim=0)
        assignment_matrix = torch.zeros_like(assignment_matrix, device=self.args.device)
        assignment_matrix[idx, torch.arange(idx.shape[0])] = 1
        backbone = assignment_matrix @ self.adj @ assignment_matrix.T
        backbone[backbone > 0] = 1
        return backbone
    
    def _update_supernode_embedding(self, lr):
        # Update supernode embedding by backbone
        euclidean_grad = self.supernode_embedding.grad
        poincare_grad = PoincareManifold.poincare_grad(euclidean_grad, self.supernode_embedding)
        self.supernode_embedding.data -= lr * poincare_grad
        self.supernode_embedding.data = self._check_norm(self.supernode_embedding.data)
        self.supernode_embedding.grad.zero_()
        
        
    def _check_norm(self, embedding, eps=1e-5):
        norm = torch.norm(embedding, dim=-1)
        
        # Keep the norm of embedding less than 1
        idx = norm > 1
        if idx.sum() > 0:
            embedding[idx] = embedding[idx] / norm[idx].unsqueeze(-1) - eps
        return embedding
    
    def _kmeans(self, adj, k, log=True):
        assert k >= 1, "k must be greater than 1"
        
        degree = adj.sum(axis=1)
        if log:
            log_degree = np.log(degree)
        
        model = cluster.KMeans(n_clusters=k, n_init='auto', max_iter=1000, random_state=0)
        model.fit(np.array(log_degree).reshape(-1, 1))
        labels = model.labels_
        log_centers = model.cluster_centers_
        
        if log:
            centers = np.exp(log_centers)

        cluster_ids = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            cluster_ids[label].append(i)
        
        return cluster_ids, centers
    
    def forward(self, tspan, X, isolate=False):
        # X: (batch_size, lookback, node_num, feature_dim)
        
        ###################
        # Identity Backbone
        ###################
        # 1. map to euclidean space from poincare space
        node_euclidean_embedding = PoincareManifold.log_map_zero(self.node_embedding)
        supernode_euclidean_embedding = PoincareManifold.log_map_zero(self.supernode_embedding)
        # 2. topology-aware representation
        node_repr = self.repr_net1(node_euclidean_embedding)
        supernode_repr = self.repr_net2(supernode_euclidean_embedding)
        # 3. assignment matrix
        assignment_prob = self.softmax(supernode_repr @ node_repr.T)
        assignment_matrix = assignment_prob
        # 4. backbone
        backbone = assignment_matrix @ self.adj @ assignment_matrix.T
        
        
        ###################
        # State aggregation
        ###################
        # 1. dynamics-aware representation
        agc_repr = self.tanh(self.agc_mlp(self.norm_lap @ X))
        # 2. state aggregation
        X_supernode = assignment_matrix @ agc_repr # batch_size, lookback, supernode_num, feature_dim
        
        
        ###################
        # Backbone Dynamics
        ###################
        # 1. predict supernode trajectory by graph neural ode
        Y_supernode = self.BackboneODE(tspan, X_supernode, backbone)  # batch_size, horizon, supernode_num, feature_dim
        # 2. copy supernode trajectory to original nodes
        Y_coarse = assignment_matrix.T @ Y_supernode # batch_size, horizon, node_num, feature_dim
        
        
        ###################
        # Refine
        ###################
        Y_refine = torch.zeros_like(Y_coarse)
        if isolate:
            Y_coarse = Y_coarse.detach()
        
        for k in range(len(self.refiners)):
            cluster_X = X[:, :, self.cluster_idx[k]]
            cluster_Y_coarse = Y_coarse[:, :, self.cluster_idx[k]]
            
            if len(self.cluster_idx[k]) == 0:
                continue
            else:
                Y_refine[:, :, self.cluster_idx[k]] = self.refiners[k](cluster_X, cluster_Y_coarse)
        
        return assignment_matrix, Y_refine, Y_supernode, (Y_coarse, X, X_supernode)
    
    def _agc_state(self, X, assignment_matrix):
        agc_repr = self.tanh(self.agc_mlp(self.norm_lap @ X))
        X_supernode = assignment_matrix @ agc_repr # batch_size, lookback, supernode_num, feature_dim
        return X_supernode
    
    def _rg_loss(self, y_rg, Y, assignment_matrix, dim=None):
                
        # Averaging Y by RG mapping M
        with torch.no_grad():
            Y_supernode = self._agc_state(Y, assignment_matrix)
    
        # MSE Loss
        if dim is None:
            rg_loss = torch.mean((y_rg - Y_supernode) ** 2)
        else:
            rg_loss = torch.mean((y_rg - Y_supernode) ** 2, dim=dim)
        
        return rg_loss, Y_supernode
    
    def _onehot_loss(self, assignment_matrix):
        entropy = -torch.sum(assignment_matrix * torch.log2(assignment_matrix + 1e-5), dim=0)
        onehot_loss = torch.mean(entropy)
        return onehot_loss
    
    def _uniform_loss(self, assignment_matrix):
        supernode_strength = torch.sum(assignment_matrix, dim=1)
        prob = supernode_strength / torch.sum(supernode_strength)
        entropy = -torch.sum(prob * torch.log2(prob + 1e-5), dim=0)
        uniform_loss = -torch.mean(entropy) # maximize entropy
        return uniform_loss
    
    def _recons_loss(self, assignment_matrix, adj):
        surrogate_adj = assignment_matrix.T @ assignment_matrix
        recons_loss = torch.norm(adj - surrogate_adj, p='fro')
        return recons_loss
    
    def _refine_loss(self, y_refine, Y, dim=None):
    
        # MSE Loss
        if dim is None:
            refine_loss = torch.mean((y_refine - Y) ** 2)
        else:
            refine_loss = torch.mean((y_refine - Y) ** 2, dim=dim)
        
        return refine_loss, Y
        
    def fit(self, train_dataloader, val_dataloader):
        
        # if os.path.exists(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model.pt'):
        #     print('Model exists, skip training')
        #     return
        # else:
        #     os.makedirs(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}', exist_ok=True)
        #     print(f'Training {self.args.model} model')
        
        optimizer = torch.optim.Adam(
            [
                {'params': self.repr_net1.parameters(), 'lr': self.args.lr},
                {'params': self.repr_net2.parameters(), 'lr': self.args.lr},
                {'params': self.agc_mlp.parameters(), 'lr': self.args.lr},
                {'params': self.BackboneODE.parameters(), 'lr': self.args.lr},
                {'params': self.refiners.parameters(), 'lr': self.args.lr},
                {'params': self.supernode_embedding, 'lr': self.args.lr},
            ]                         
            , lr=self.args.lr)
        scheduler = StepLR(optimizer, step_size=self.args.lr_step, gamma=self.args.lr_decay)
        
        dt = self.args[self.args.dynamics].dt
        start_t = (self.args.lookback) * dt
        end_t = (self.args.lookback + self.args.horizon - 1) * dt
        tspan = torch.linspace(start_t, end_t, self.args.horizon).to(self.args.device)
        
        train_loss_list, val_loss_list = [], []
        for epoch in range(1, self.args.max_epoch+1):
            train_loss = 0.0
            self.train()
            for i, (X, Y) in enumerate(train_dataloader):
                assignment_matrix, y_refine, y_rg, _ = self(tspan, X)
                rg_loss, _ = self._rg_loss(y_rg, Y, assignment_matrix)
                refine_loss, _ = self._refine_loss(y_refine, Y)
                onehot_loss = self._onehot_loss(assignment_matrix)
                uniform_loss = self._uniform_loss(assignment_matrix)
                recons_loss = self._recons_loss(assignment_matrix, self.adj)
                loss = refine_loss + rg_loss + onehot_loss + recons_loss + uniform_loss
                
                optimizer.zero_grad()
                loss.backward()
                self.supernode_embedding.grad = PoincareManifold.poincare_grad(self.supernode_embedding.grad, self.supernode_embedding) # rescale euclidean grad to poincare grad
                optimizer.step()
                
                with torch.no_grad():
                    self.supernode_embedding.data = self._check_norm(self.supernode_embedding.data)
                
                train_loss += loss.item()
            print(f'\rEpoch[{epoch}/{self.args.max_epoch}] train backbone: {rg_loss.item():.4f}, refine: {refine_loss.item():.4f}, onehot: {onehot_loss.item():.4f}, recons: {recons_loss.item():.4f}, uniform: {uniform_loss.item():.4f}', end='')
            train_loss_list.append([epoch, train_loss / len(train_dataloader)])
            
            scheduler.step()
            if epoch % self.args.val_interval == 0:
                self.eval()
                val_loss = 0
                for i, (X, Y) in enumerate(val_dataloader):
                    assignment_matrix, y_refine, y_rg, info = self(tspan, X) # info: (Y_coarse, X_reindex, Y_coarse, X_rg, kappa_reindex)
                    rg_loss, Y_coarse = self._rg_loss(y_rg, Y, assignment_matrix)
                    refine_loss, Y_reindex = self._refine_loss(y_refine, Y)
                    onehot_loss = self._onehot_loss(assignment_matrix)
                    uniform_loss = self._uniform_loss(assignment_matrix)
                    recons_loss = self._recons_loss(assignment_matrix, self.adj)
                    loss = refine_loss
                    val_loss += loss.item()
                    
                    if i == 0:
                        os.makedirs(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}', exist_ok=True)
                        drawTraj(y_rg[:,:,:100], Y_coarse[:,:,:100], 'pred', 'true', dim=0, out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/rg_pred.png')
                        drawTraj(info[2][:,:,:10], Y_coarse[:,:12,:10], 'rg_x', 'rg_y', dim=0, out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/rg_traj.png')
                        drawTraj(info[1][:,:,:20], info[2][:,:,:10], 'x', 'x_rg', dim=0, out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/x_rg.png')
                        drawTraj(y_refine[:,:,:100], info[0][:,:,:100], 'refined', 'coarse', dim=0, out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine.png')
                        drawTraj(y_refine[:,:,:100], Y_reindex[:,:,:100], 'pred', 'true', dim=0, out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/result.png')
                        drawTraj(Y_reindex[:,:,:100], Y_coarse[:,:,:50], 'Y', 'Y_coarse', dim=0, out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/y_rg.png')
                        
                        # Draw the backbone
                        idx = torch.argmax(assignment_matrix, dim=0)
                        assignment_matrix = torch.zeros_like(assignment_matrix, device=self.args.device)
                        assignment_matrix[idx, torch.arange(idx.shape[0])] = 1
                        backbone = assignment_matrix @ self.adj @ assignment_matrix.T
                        # backbone[backbone > 0] = 1
                        draw_embedding(backbone, self.supernode_embedding, f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/supernode_poincare.png')
                        
                        # Assignment distribution
                        count = torch.sum(assignment_matrix, dim=1)
                        valid_num = len(count[count > 0])
                
                print(f'\nEpoch[{epoch}/{self.args.max_epoch}] val backbone: {rg_loss.item():.4f}, refine: {refine_loss.item():.4f}, onehot: {onehot_loss.item():.4f}, recons: {recons_loss.item():.4f}, uniform: {uniform_loss.item():.4f} | assignment: {valid_num}/{self.supernode_embedding.shape[0]}')
                val_loss_list.append([epoch, val_loss / len(val_dataloader)])
            
                # Save model
                torch.save(self.state_dict(), f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model_{epoch}.pt')
                
        # Draw loss curve
        train_loss_list = np.array(train_loss_list)
        val_loss_list = np.array(val_loss_list)
        plt.figure(figsize=(5, 4))
        plt.plot(train_loss_list[:, 0], train_loss_list[:, 1], label='train')
        plt.plot(val_loss_list[:, 0], val_loss_list[:, 1], label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/loss.png', dpi=300)
        
        # Save model
        torch.save(self.state_dict(), f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model.pt')
        
        # Fine-tune refiner
        self.refine(train_dataloader, val_dataloader)
        
        # Release memory
        del train_dataloader, val_dataloader, optimizer, scheduler
    
    def refine(self, train_dataloader, val_dataloader):
        
        dt = self.args[self.args.dynamics].dt
        start_t = (self.args.lookback) * dt
        end_t = (self.args.lookback + self.args.horizon - 1) * dt
        tspan = torch.linspace(start_t, end_t, self.args.horizon).to(self.args.device)
        
        optimizer = torch.optim.Adam(self.refiners.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=self.args.lr_step, gamma=self.args.lr_decay)
        
        for epoch in range(1, 10+1):
            train_loss = 0.0
            self.train()
            for i, (X, Y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                _, y_refine, y_rg, _ = self(tspan, X, isolate=True)
                refine_loss, _ = self._refine_loss(y_refine, Y)
                loss = refine_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f'\rEpoch[{epoch}] train refine loss: {refine_loss.item():.4f}', end='')
            
            scheduler.step()
            if epoch % self.args.val_interval == 0:
                self.eval()
                val_loss = 0
                for i, (X, Y) in enumerate(val_dataloader):
                    _, y_refine, y_rg, info = self(tspan, X, isolate=True) # info: (Y_coarse, X_reindex, Y_coarse, X_rg, kappa_reindex)
                    refine_loss, Y_reindex = self._refine_loss(y_refine, Y)
                    loss = refine_loss
                    val_loss += loss.item()
                    if i == 0:
                        os.makedirs(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine', exist_ok=True)
                        drawTraj(info[1][:,:,:20], info[2][:,:,:10], 'x', 'x_rg', out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine/x_rg.png')
                        drawTraj(y_refine[:,:,:100], info[0][:,:,:100], 'refined', 'coarse', out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine/refine.png')
                        drawTraj(y_refine[:,:,:100], Y_reindex[:,:,:100], 'pred', 'true', out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/epoch_{epoch}/refine/result.png')
                print(f'\nEpoch[{epoch}/10] val refine loss: {refine_loss.item():.4f}')
                
        # Save model
        torch.save(self.state_dict(), f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model_refine.pt')
    
    def test(self, test_dataloader):
                
        # Load model
        try:
            self.load_state_dict(torch.load(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model_refine.pt'))
        except:
            self.load_state_dict(torch.load(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/model.pt'))
        self.to(self.args.device)
        
        dt = self.args[self.args.dynamics].dt
        start_t = (self.args.lookback) * dt
        end_t = (self.args.lookback + self.args.horizon - 1) * dt
        tspan = torch.linspace(start_t, end_t, self.args.horizon).to(self.args.device)
        
        # Test
        self.eval()
        print('Testing...')
        ground_truth = np.zeros((len(test_dataloader), self.args.batch_size, self.args.horizon, self.args.node_num, self.feature_dim))
        predict = np.zeros((len(test_dataloader), self.args.batch_size, self.args.horizon, self.args.node_num, self.feature_dim))
        for i, (X, Y) in enumerate(test_dataloader):
            assignment_matrix, y_refine, y_rg, info = self(tspan, X)
            
            if i == len(test_dataloader)-1:
                Y_coarse = self._agc_state(Y, assignment_matrix)
                os.makedirs(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test', exist_ok=True)
                drawTraj(y_rg[:,:,:100], Y_coarse[:,:,:100], 'pred', 'true', out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/rg_pred.png', num=3)
                drawTraj(info[2][:,:,:10], Y_coarse[:,:12,:10], 'rg_x', 'rg_y', dim=0, out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/rg_traj.png')
                drawTraj(info[1][:,:,:20], info[2][:,:,:10], 'x', 'x_rg', dim=0, out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/x_rg.png')
                drawTraj(y_refine[:,:,:200], info[0][:,:,:200], 'refined', 'coarse', out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/refine.png')
                drawTraj(y_refine[:,:,:200], Y[:,:,:200], 'pred', 'true', out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/result.png', num=3)
                drawTraj(Y[:,:,:200], Y_coarse[:,:,:100], 'Y', 'Y_coarse', out_path=f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/y_rg.png')
            
            ground_truth[i, :, :, :, :] = Y.cpu().detach().numpy()
            predict[i, :, :, :, :] = y_refine.cpu().detach().numpy()
            
        backbone_pred = y_rg.cpu().detach().numpy()
        backbone_true = Y_coarse.cpu().detach().numpy()
        
        # Draw the backbone
        idx = torch.argmax(assignment_matrix, dim=-1)
        assignment_matrix = torch.zeros_like(assignment_matrix)
        assignment_matrix[torch.arange(idx.shape[0]), idx] = 1
        backbone = assignment_matrix @ self.adj @ assignment_matrix.T
        # backbone[backbone > 0] = 1
        draw_embedding(backbone, self.supernode_embedding, f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/test/supernode_poincare.png')
        
        # Save result
        time_cost = 0.0
        ground_truth = ground_truth.reshape(-1, self.args.horizon, self.args.node_num, self.feature_dim)
        predict = predict.reshape(-1, self.args.horizon, self.args.node_num, self.feature_dim)
        np.savez(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/backbone_result.npz', backbone_pred=backbone_pred, backbone_true=backbone_true)
        np.savez(f'{self.args.log_dir}/{self.args.dynamics}/{self.args.model}/result.npz', ground_truth=ground_truth, predict=predict, time_cost=time_cost)
        