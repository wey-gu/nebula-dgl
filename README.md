# nebula-dgl

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

nebula-dgl is the Lib for Nebula Graph integration with Deep Graph Library (DGL).

# Guide

## Installation

```bash
# this is needed until nebula-python release to fix storageclient issue by including https://github.com/vesoft-inc/nebula-python/pull/219
python3 -m pip install git+https://github.com/vesoft-inc/nebula-python.git@8c328c534413b04ccecfd42e64ce6491e09c6ca8

# build and install
python3 -m pip install .
```

## Playground

Clone this repository to your local directory first.

```bash
git clone https://github.com/wey-gu/nebula-dgl.git
cd nebula-dgl
```

0. Deploy NebulaGraph playground with Nebula-UP:

Install NebulaGraph:

```bash
curl -fsSL nebula-up.siwei.io/install.sh | bash
```

Load example data:

```bash
~/.nebula-up/load-basketballplayer-dataset.sh
```

1. Create a jupyter notebook in same docker network: `nebula-net`

```bash
docker run -it --name dgl -p 8888:8888 --network nebula-net \
    -v "$PWD":/home/jovyan/work jupyter/datascience-notebook \
    start-notebook.sh --NotebookApp.token='nebulagraph'
```
Now you can either access the notebook at `http://localhost:8888/?token=nebulagraph`

or run ipython with the container:

```bash
docker exec -it dgl ipython
```


2. Install nebula-dgl in notebook:

```bash
cd work
!python3 -m pip install git+https://github.com/vesoft-inc/nebula-python.git@8c328c534413b04ccecfd42e64ce6491e09c6ca8
!python3 -m pip install .
```

3. Try with a homogeneous graph:

```python
import yaml

nebula_config = {
    "graph_hosts": [
                ('graphd', 9669),
                ('graphd1', 9669),
                ('graphd2', 9669)
            ],
    "user": "root",
    "password": "nebula",
}

import networkx as nx


with open('example/homogeneous_graph_example.yaml', 'r') as f:
    feature_mapper = yaml.safe_load(f)

nebula_loader = NebulaLoader(nebula_config, feature_mapper)
homo_dgl_graph = nebula_loader.load()
nx.draw(nx_g, with_labels=True, pos=nx.spring_layout(nx_g))
```

Result:

![nx_draw](https://user-images.githubusercontent.com/1651790/181154556-c25532f9-33ff-4cc8-85d9-62cb559d7f1a.png)

4. Compute the degree centrality of the graph:

```python
nx.degree_centrality(nx_g)
```
Result:

```python
{0: 0.0,
 1: 0.038461538461538464,
 2: 0.019230769230769232,
 3: 0.019230769230769232,
 4: 0.057692307692307696,
 5: 0.057692307692307696,
 6: 0.038461538461538464,
 7: 0.25,
 8: 0.19230769230769232,
 9: 0.0,
 10: 0.019230769230769232,
 11: 0.038461538461538464,
 12: 0.038461538461538464,
 13: 0.038461538461538464,
 14: 0.09615384615384616,
 15: 0.038461538461538464,
 16: 0.0,
 17: 0.09615384615384616,
 18: 0.038461538461538464,
 19: 0.038461538461538464,
 20: 0.0,
 21: 0.0,
 22: 0.038461538461538464,
 23: 0.019230769230769232,
 24: 0.019230769230769232,
 25: 0.0,
 26: 0.038461538461538464,
 27: 0.057692307692307696,
 28: 0.0,
 29: 0.019230769230769232,
 30: 0.0,
 31: 0.038461538461538464,
 32: 0.11538461538461539,
 33: 0.0,
 34: 0.038461538461538464,
 35: 0.21153846153846156,
 36: 0.13461538461538464,
 37: 0.09615384615384616,
 38: 0.038461538461538464,
 39: 0.13461538461538464,
 40: 0.09615384615384616,
 41: 0.019230769230769232,
 42: 0.13461538461538464,
 43: 0.07692307692307693,
 44: 0.09615384615384616,
 45: 0.11538461538461539,
 46: 0.11538461538461539,
 47: 0.07692307692307693,
 48: 0.11538461538461539,
 49: 0.019230769230769232,
 50: 0.038461538461538464,
 51: 0.11538461538461539,
 52: 0.057692307692307696}
 ```

## Nebula Graph to DGL

```python
from nebula_dgl import NebulaLoader


nebula_config = {
    "graph_hosts": [
                ('graphd', 9669),
                ('graphd1', 9669),
                ('graphd2', 9669)
            ],
    "user": "root",
    "password": "nebula",
}

# load feature_mapper from yaml file
with open('example/nebula_to_dgl_mapper.yaml', 'r') as f:
    feature_mapper = yaml.safe_load(f)

nebula_loader = NebulaLoader(nebula_config, feature_mapper)
dgl_graph = nebula_loader.load()

```

## Play homogeneous graph algorithms in networkx

```python

import networkx

with open('example/homogeneous_graph.yaml', 'r') as f:
    feature_mapper = yaml.safe_load(f)

nebula_loader = NebulaLoader(nebula_config, feature_mapper)
homo_dgl_graph = nebula_loader.load()
nx_g = homo_dgl_graph.to_networkx()

# plot it
networkx.draw(nx_g, with_lables=True)

# get degree
networkx.degree(nx_g)

# get degree centrality
networkx.degree_centrality(nx_g)
```
