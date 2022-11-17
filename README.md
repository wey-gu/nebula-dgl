# nebula-dgl

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

nebula-dgl is the Lib for Nebula Graph integration with Deep Graph Library (DGL).

> nebula-dgl is still WIP, there is a demo project [here](https://github.com/wey-gu/NebulaGraph-Fraud-Detection-GNN/) .

# Guide

## Installation

```bash
# this is needed until nebula-python release to fix storageclient issue by including https://github.com/vesoft-inc/nebula-python/pull/219
python3 -m pip install nebula3-python==3.3.0
python3 -m pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html

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
    start-notebook.sh --NotebookApp.token='secret'
```
Now you can either:

- access the notebook at http://localhost:8888/lab/tree/work?token=secret and create a new notebook.

Or:

- run ipython with the container:

```bash
docker exec -it dgl ipython
cd work
```


2. Install nebula-dgl in notebook:

Install nebula-dgl:

```
!python3 -m pip install python3 -m pip install nebula3-python==3.3.0
!python3 -m pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
!python3 -m pip install .
```

3. Try with a homogeneous graph:

```python
import yaml
import networkx as nx

from nebula_dgl import NebulaLoader


nebula_config = {
    "graph_hosts": [
                ('graphd', 9669),
                ('graphd1', 9669),
                ('graphd2', 9669)
            ],
    "nebula_user": "root",
    "nebula_password": "nebula",
}

with open('example/homogeneous_graph.yaml', 'r') as f:
    feature_mapper = yaml.safe_load(f)

nebula_loader = NebulaLoader(nebula_config, feature_mapper)
homo_dgl_graph = nebula_loader.load()

nx_g = homo_dgl_graph.to_networkx()
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
 1: 0.04,
 2: 0.02,
 3: 0.02,
 4: 0.06,
 5: 0.06,
 6: 0.04,
 7: 0.24,
 8: 0.16,
 9: 0.0,
 10: 0.02,
 11: 0.04,
 12: 0.04,
 13: 0.04,
 14: 0.1,
 15: 0.04,
 16: 0.0,
 17: 0.1,
 18: 0.04,
 19: 0.04,
 20: 0.0,
 21: 0.0,
 22: 0.04,
 23: 0.02,
 24: 0.02,
 25: 0.04,
 26: 0.06,
 27: 0.0,
 28: 0.02,
 29: 0.0,
 30: 0.04,
 31: 0.12,
 32: 0.04,
 33: 0.22,
 34: 0.14,
 35: 0.1,
 36: 0.04,
 37: 0.14,
 38: 0.1,
 39: 0.02,
 40: 0.14,
 41: 0.08,
 42: 0.1,
 43: 0.12,
 44: 0.12,
 45: 0.08,
 46: 0.1,
 47: 0.02,
 48: 0.04,
 49: 0.12,
 50: 0.06}
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
    "nebula_user": "root",
    "nebula_password": "nebula"
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
