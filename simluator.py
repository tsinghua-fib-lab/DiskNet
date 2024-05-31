import os
import pickle
import numpy as np
import urllib.request
import zipfile
import networkx as nx
from sdeint import itoEuler

from utils import HyperbolicEmbedding
from dynamics import *


sites = {
    'Drosophila': 'https://nrvis.com/download/data/bn/bn-fly-drosophila_medulla_1.zip',
    'PowerGrid': 'https://nrvis.com/download/data/power/power-bcspwr10.zip',
    'Social': 'https://nrvis.com/download/data/soc/fb-pages-tvshow.zip',
    'Web': 'https://nrvis.com/download/data/web/web-EPA.zip',
    'Airport': 'https://nrvis.com/download/data/inf/inf-openflights.zip',
}


class NetworkSimulator(object):

    def __init__(self, args: dict):
        self.args = args
        self.G = None
        self.HyperbolicG = None

    def buildNetwork(self):
        try:
            with open(f'{self.args.data_dir}/graph.pkl', 'rb') as file:
                graph = pickle.load(file)
                self.G = graph.G
                print(f'Load {self.args.graph_type} graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges')
        
        except:
            os.makedirs(self.args.data_dir, exist_ok=True)
            
            # Create graph
            if self.args.graph_type == 'BA':
                self.G = nx.barabasi_albert_graph(self.args.node_num, self.args.edge_num, seed=self.args.seed)
            elif self.args.graph_type == 'WS':
                self.G = nx.watts_strogatz_graph(self.args.node_num, self.args.ring_lattice_k, self.args.rewiring_prob, seed=self.args.seed)
            elif self.args.graph_type == 'Drosophila':
                self._downloadNetwork()
                self.G = nx.read_edgelist(f'{self.args.data_dir}/bn/bn-fly-drosophila_medulla_1.edges', create_using=nx.DiGraph)
                self.G = self.G.to_undirected()
            elif self.args.graph_type == 'PowerGrid':
                self._downloadNetwork()
                self.G = nx.read_edgelist(f'{self.args.data_dir}/power-bcspwr10.mtx', create_using=nx.DiGraph)
                self.G = self.G.to_undirected()
            elif self.args.graph_type == 'Social':
                self._downloadNetwork()
                self.G = nx.read_edgelist(f'{self.args.data_dir}/fb-pages-tvshow.edges', create_using=nx.DiGraph)
                self.G = self.G.to_undirected()
            elif self.args.graph_type == 'Web':
                self._downloadNetwork()
                self.G = nx.read_edgelist(f'{self.args.data_dir}/web-EPA.edges', create_using=nx.DiGraph)
                self.G = self.G.to_undirected()
            elif self.args.graph_type == 'Airport':
                self._downloadNetwork()
                self.G = nx.read_edgelist(f'{self.args.data_dir}/inf-openflights.edges', create_using=nx.DiGraph)
                self.G = self.G.to_undirected()
            else:
                raise Exception(f'Invalid graph type: {self.args.graph_type}')
            
            # delete isolated nodes
            self.G.remove_nodes_from(list(nx.isolates(self.G)))
            # delete self-loop edges
            self.G.remove_edges_from(nx.selfloop_edges(self.G))
            # only keep the largest connected component
            self.G = self.G.subgraph(max(nx.connected_components(self.G), key=len))
            # relabel nodes
            self.G = nx.convert_node_labels_to_integers(self.G, first_label=0)
            # keep even number of nodes for static RG model
            if self.G.number_of_nodes() % 2 == 1:
                print(self.G.number_of_nodes())
                self.G.remove_node(np.random.choice(list(self.G.nodes)))
                print(self.G.number_of_nodes())
            # relabel nodes
            self.G = nx.convert_node_labels_to_integers(self.G, first_label=0)
            
            with open(f'{self.args.data_dir}/graph.pkl', 'wb') as file:
                pickle.dump(self, file)
            
            nx.write_edgelist(self.G, f'{self.args.data_dir}/graph.txt', data=False)
            
            print(f'Save {self.args.graph_type} graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges')
            
        return self.G, nx.to_numpy_array(self.G)
    
    def getHyperbolicEmbedding(self):
        he = HyperbolicEmbedding(args=self.args)
        s1_kappa, s1_angular, h1_radius, mu, beta, radius_s1 = he.fit_transform()
        
        exp_degree, he_degree = np.mean([d for n, d in nx.degree(self.G)]), 0.1
        # while abs(exp_degree - he_degree) > 0.1:
        p_matrix = self._connectivity_probability_matrix(s1_kappa, s1_angular, mu, beta, radius_s1)
        sampled_A = np.random.binomial(1, p_matrix)
        self.HyperbolicG = nx.from_numpy_array(sampled_A)
        he_degree = np.mean([d for n, d in nx.degree(self.HyperbolicG)])
        mu = mu * (exp_degree / he_degree)
        
        nx.write_edgelist(self.HyperbolicG, f'{self.args.log_dir}/HE/hyperbolic_graph.txt', data=False)
        
        print(f'clustering coeficient: {nx.average_clustering(self.G):.3f}-->{nx.average_clustering(self.HyperbolicG):.3f}')
        print(f'degree: {np.mean([d for n, d in nx.degree(self.G)]):.3f}-->{np.mean([d for n, d in nx.degree(self.HyperbolicG)]):.3f}')

        return s1_kappa, s1_angular, h1_radius, mu, beta, radius_s1
    
    def getSimTraj(self):
        
        try:
            with np.load(f'{self.args.data_dir}/{self.args.dynamics}/dynamics.npz') as data:
                X = data['X']
            print(f'Load {self.args.dynamics} dynamics with {self.args.node_num} nodes and {self.args[self.args.dynamics].total_t} time steps')
        
        except:
            dim = self.args[self.args.dynamics].dim
            
            if self.args.dynamics == 'HindmarshRose':
                sde = HindmarshRose(args=self.args, A=nx.to_numpy_array(self.G))
                x0_1 = np.random.uniform(-1, 0, size=self.args.node_num)
                x0_2 = np.random.uniform(-5, 0, size=self.args.node_num)
                x0_3 = np.random.uniform(3, 3.5, size=self.args.node_num)
                x0 = np.concatenate((x0_1, x0_2, x0_3))
            elif self.args.dynamics == 'FitzHughNagumo':
                sde = FitzHughNagumo(args=self.args, A=nx.to_numpy_array(self.G))
                x0 = np.random.uniform(-1, 1, size=self.args.node_num*dim)
            elif self.args.dynamics == 'CoupledRossler':
                sde = CoupledRossler(args=self.args, A=nx.to_numpy_array(self.G))
                x0 = np.random.uniform(-0.05, 0.05, size=self.args.node_num*dim)
                
            tspan = np.arange(0, self.args[self.args.dynamics].total_t, self.args[self.args.dynamics].sim_dt)
            sol = itoEuler(sde.f, sde.g, x0, tspan) # (total_t, node_num*feature_dim)
            
            # downsample
            ratio = int(self.args[self.args.dynamics].dt / self.args[self.args.dynamics].sim_dt)
            sol = sol[::ratio]
            
            X = np.zeros((sol.shape[0], self.args.node_num, dim))
            for i in range(dim):
                X[:, :, i] = sol[:, i*self.args.node_num:(i+1)*self.args.node_num]
            
            if self.args.dynamics == 'CoupledKuramoto':
                X = np.sin(X)
            
            os.makedirs(f'{self.args.data_dir}/{self.args.dynamics}', exist_ok=True)
            np.savez(f'{self.args.data_dir}/{self.args.dynamics}/dynamics.npz', X=X)
            print(f'Save {self.args.dynamics} dynamics with {self.args.node_num} nodes and {self.args[self.args.dynamics].total_t} time steps')
        
        return X
    
    def _downloadNetwork(self):
        url = sites[self.args.graph_type]
        file_name = f"{self.args.data_dir}/download.zip"
        urllib.request.urlretrieve(url, file_name)

        with zipfile.ZipFile(file_name, "r") as zip_ref:
            zip_ref.extractall(self.args.data_dir)
        
        os.remove(file_name)
        
        if self.args.graph_type == 'PowerGrid':
            # delete header
            with open(f'{self.args.data_dir}/power-bcspwr10.mtx', 'r') as f:
                lines = f.readlines()
            with open(f'{self.args.data_dir}/power-bcspwr10.mtx', 'w') as f:
                f.writelines(lines[14:])
        elif self.args.graph_type == 'Social':
            # replace ',' with ' '
            with open(f'{self.args.data_dir}/fb-pages-tvshow.edges', 'r') as f:
                lines = f.readlines()
            with open(f'{self.args.data_dir}/fb-pages-tvshow.edges', 'w') as f:
                for line in lines:
                    f.write(line.replace(',', ' '))
        elif self.args.graph_type == 'Airport':
            # delete header
            with open(f'{self.args.data_dir}/inf-openflights.edges', 'r') as f:
                lines = f.readlines()
            with open(f'{self.args.data_dir}/inf-openflights.edges', 'w') as f:
                f.writelines(lines[2:])
    
    def _connectivity_probability_matrix(self, kappa, angular, mu, beta, radius):
        """p_ij = 1 / (1 + (radius*(delta_angular) / (mu*kappa_i*kappa_j))**beta)"""
        
        abs_delta_angular = np.abs(angular.reshape(-1, 1) - angular.reshape(1, -1))
        delta_angular = np.minimum(abs_delta_angular, 2 * np.pi - abs_delta_angular)
        kappa_mul = kappa.reshape(-1, 1) * kappa.reshape(1, -1)
        rescaled_dist = radius * delta_angular / (mu * kappa_mul)
        p_matrix = 1 / (1 + rescaled_dist ** beta) - np.eye(self.args.node_num)
        
        return p_matrix        
    