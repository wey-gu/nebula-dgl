# nebula-dgl

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

nebula-dgl is the Lib for Nebula Graph integration with Deep Graph Library (DGL).

# Guide

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
