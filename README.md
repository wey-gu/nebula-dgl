# nebula-dgl

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

nebula-dgl is the Lib for Nebula Graph integration with Deep Graph Library (DGL).

# Guide

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