from omegaconf import OmegaConf
import warnings; warnings.filterwarnings('ignore')

from model import DiskNet
from data import Dataset
from simluator import NetworkSimulator
from utils import *


def main():
    # Read config
    conf = OmegaConf.load('config.yaml')
    
    # Set random seed and cpu number
    set_cpu_num(conf.cpu_num)
    seed_everything(conf.seed)
    
    # Set data and log directory
    if conf.graph_type in ['BA', 'WS']:
        conf.data_dir = f'data/{conf.graph_type}_n{conf.node_num}_{conf.seed}'
        conf.log_dir = f'logs/{conf.graph_type}_n{conf.node_num}_{conf.seed}/l{conf.lookback}_h{conf.horizon}'
    else:
        conf.data_dir = f'data/{conf.graph_type}'
        conf.log_dir = f'logs/{conf.graph_type}/l{conf.lookback}_h{conf.horizon}'
    
    # Create graph
    simulator = NetworkSimulator(args=conf)
    network, adj_matrix = simulator.buildNetwork()
    
    # Draw graph
    conf.node_num = network.number_of_nodes()
    drawGraph(network, layout='string', filter='random', threshold=0.5, out_path=f'{conf.data_dir}/graph.png')
    
    # Simulate network dynamics
    simulator.getSimTraj()
    
    # Dataset
    train_dataset = Dataset(conf, mode='train')
    val_dataset = Dataset(conf, mode='val')
    test_dataset = Dataset(conf, mode='test')
    
    # Model
    model = DiskNet(conf, adj_matrix)
    print_model_summary(model)
    
    # Train
    model.fit(train_dataset.getLoader(), val_dataset.getLoader())
    
    # Test
    model.test(test_dataset.getLoader())
    

if __name__ == '__main__':
    
    main()