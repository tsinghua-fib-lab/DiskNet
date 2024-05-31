# DiskNet

The repo is the official implementation for our  paper: "Predicting Long-term Dynamics of Complex Networks via Identifying Skeleton in Hyperbolic Space” (KDD 2024). 



## Overall Architecture

DiskNet: (1) Hyperbolic Renormalization Group, which identifies the representation and skeleton of network dynamics; (2) Neural Dynamics on Skeleton, which models the dynamics of super-nodes on the skeleton; and (3) Degree-based Super-Resolution, which lifts the predicted values of super-nodes to the original nodes.

![image-20240531124436518](https://my-picture-1311448338.cos.ap-nanjing.myqcloud.com/SiliconLRK/image-20240531124436518.png)



## Environment Setup

```
conda create --name <env> --file requirement.txt
```



## Usage

**Config:**

graph_type: `BA `, `WS`, `Drosophila`, `Social`, `Web`, `PowerGrid` or `Airport`;

dynamics: `HindmarshRose`, `FitzHughNagumo` or `CoupledRossler`

**Run:**

```shell
python main.py
```



## Citation

If you find this repo helpful, please cite our paper.

```
Coming soon.
```
