import os
import torch
import random
import mercator
import numpy as np
import networkx as nx
import scienceplots
import matplotlib.pyplot as plt

from metrics import *


plt.style.use(['ieee', 'science', 'no-latex'])
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams["font.family"] = 'Arial'


def set_cpu_num(cpu_num: int = 1):
    if cpu_num <= 0: return
    
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def seed_everything(seed: int = 42):
    # Set the random seed for Python's built-in random module
    random.seed(seed)
    
    # Set the random seed for NumPy
    np.random.seed(seed)
    
    # Set the random seed for torch operations
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    print(model)
    print("Number of parameters: {:,}".format(count_parameters(model)))

    print("\nParameter details:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name, parameter.shape, parameter.device, parameter.dtype, parameter.numel())


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
        radius_s1 = float(lines[10].split()[-1])
        radius_h2 = float(lines[11].split()[-1])
        
        s1_kappa = np.zeros(node_num)
        s1_angular = np.zeros(node_num)
        h2_radius = np.zeros(node_num)
        for i in range(15, 15+node_num):
            s1_kappa[i-15] = float(lines[i].split()[1])
            s1_angular[i-15] = float(lines[i].split()[2])
            h2_radius[i-15] = float(lines[i].split()[3])
        
        return s1_kappa, s1_angular, h2_radius, mu, beta, radius_s1


def drawGraph(G: nx.Graph, layout: str = 'random', filter: str = 'none', threshold = 0.3, out_path: str = 'graph.png'):
    """
    param G: networkX graph
    param layout: layout type, options: random, circular, spring, spectral, shell, kamada_kawai (default: random)
    param filter: disparity filter, options: none, degree, betweenness, random (default: none)
    param threshold: disparity filter threshold (default: 0.3)
    """
    
    # disparity filter
    if filter == 'degree':
        degree_centrality = nx.degree_centrality(G)
        core_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]
        G = G.subgraph(core_nodes)
    elif filter == 'betweenness':
        betweenness_centrality = nx.betweenness_centrality(G)
        core_nodes = [node for node, centrality in betweenness_centrality.items() if centrality >= threshold]
        G = G.subgraph(core_nodes)
    elif filter == 'random':
        core_nodes = [node for node in G.nodes() if np.random.rand() >= threshold]
        G = G.subgraph(core_nodes)
    else:
        pass
    
    # layout
    if layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.random_layout(G)
    
    # node color and size
    node_color = [G.degree(v) for v in G]
    node_size = [v * 10 for v in node_color]
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_color=node_color, node_size=node_size, edge_color=(0, 0, 0, 0.25))
    plt.savefig(out_path, dpi=300)
    

def drawTraj(X1, X2, title1, title2, yticks1=None, yticks2=None, dim=0, out_path='traj.png', num=2):
    # X1, X2: (Batch, Node Num, Horizon)
    if isinstance(X1, torch.Tensor):
        data1 = X1[0,:,:,dim].detach().cpu().numpy().T
        data2 = X2[0,:,:,dim].detach().cpu().numpy().T
    else:
        data1 = X1[0,:,:,dim].T
        data2 = X2[0,:,:,dim].T
    if yticks1 is not None:
        yticks1 = yticks1.detach().cpu().numpy()[:,0]
        yticks2 = yticks2.detach().cpu().numpy()[:,0]
    
    zmax = max(np.max(data1), np.max(data2))
    zmin = min(np.min(data1), np.min(data2))
    
    # Heatmap
    plt.figure(figsize=(num*4, 4))
    plt.subplot(1, num, 1)
    plt.imshow(data1, cmap='hot', interpolation='nearest', vmin=zmin, vmax=zmax)
    plt.ylabel('Node')
    plt.xlabel('Time Step')
    if yticks1 is not None:
        plt.yticks(np.arange(0, len(yticks1), 1), yticks1)
    plt.title(title1)
    plt.subplot(1, num, 2)
    plt.imshow(data2, cmap='hot', interpolation='nearest', vmin=zmin, vmax=zmax)
    plt.ylabel('Node')
    plt.xlabel('Time Step')
    if yticks1 is not None:
        plt.yticks(np.arange(0, len(yticks2), 1), yticks2)
    plt.title(title2)
    if num == 3:
        plt.subplot(1, num, 3)
        plt.imshow(np.abs(data2-data1), cmap='hot', interpolation='nearest', vmin=zmin, vmax=zmax)
        plt.ylabel('Node')
        plt.xlabel('Time Step')
        if yticks1 is not None:
            plt.yticks(np.arange(0, len(yticks2), 1), yticks2)
        plt.title('MAE')
    # plt.colorbar()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    
    
def draw_embedding(adj, embedding, out_path, label=None, mask=None):
    """
    Draw the embedding.
    """
    if os.path.exists(out_path):
        return
    
    embedding = embedding.cpu().detach().numpy()
    size = adj.diagonal().cpu().detach().numpy() + 1
    
    plt.figure(figsize=(4, 4))
    
    # edge
    tmp = adj >= 1
    idx = np.where(tmp.cpu().detach().numpy())
    for i in range(len(idx[0])):
        plt.plot([embedding[idx[0][i], 0], embedding[idx[1][i], 0]], [embedding[idx[0][i], 1], embedding[idx[1][i], 1]], color="#555555", alpha=0.03, linewidth=0.5)
        
    # node
    if label is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=8, alpha=1.0)
    else:
        if mask is None:
            mask = np.ones_like(label, dtype=bool)
        else:
            mask = mask.cpu().detach().numpy()

        for i in np.unique(label):
            mask_ = (label == i) & mask
            plt.scatter(embedding[mask_, 0], embedding[mask_, 1], s=size[mask_]*1.2, alpha=1.0, label=f'Supernode {i}')
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    # plt.legend(loc='upper right', fontsize=8, frameon=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def StatisticGraph():
    
    import pickle
    import networkx as nx
    for graph_type in ['PowerGrid', 'Drosophila', 'Social', 'Web', 'Airport', 'BA_n5000_612', 'WS_n5000_612']:
        with open(f'data/{graph_type}/graph.pkl', 'rb') as file:
            graph = pickle.load(file).G # nx.Graph
        
        node_num = graph.number_of_nodes()
        edge_num = graph.number_of_edges()
        avg_degree = sum(dict(graph.degree()).values()) / node_num
        avg_clustering = nx.average_clustering(graph)
        density = nx.density(graph)
        print(f'{graph_type.rjust(12)}: Node Num={node_num}, Edge Num={edge_num}, Avg Degree={avg_degree:.4f}, Avg Clustering={avg_clustering:.4f}, Density={density:.4f}')


# if __name__ == '__main__':
#     StatisticGraph()