from nebula3.mclient import MetaCache
from nebula3.sclient.GraphStorageClient import GraphStorageClient

from typing import Dict, List, Callable, Tuple

import logging


logger = logging.getLogger(__package__)
ENUMERATION_DEFAULT_VALUE = -1
NEBULA_TYPE_MAP = {
    "int": "as_int",
    "double": "as_double",
    "str": "as_string",
    "float": "as_double",
}


class NebulaPartLoader():
    """
    NebulaPartLoader is a class that loads vertex and edge data from Nebula Graph Cluster.

    """

    def __init__(self, part_id: int, meta_hosts: List, edge_feature_dict: Dict, tag_feature_dict: Dict):
        """
        Initialize the NebulaLoader class.
        """
        self.part_id = part_id
        self.edge_feature_dict = edge_feature_dict
        self.tag_feature_dict = tag_feature_dict
        self.meta_hosts = meta_hosts

    def get_storage_client(self):
        """
        Get the GraphStorageClient.
        """
        meta_cache = MetaCache(self.meta_hosts, 50000)
        storage_client = GraphStorageClient(meta_cache)
        return storage_client

    def get_feature_transform_function(self, features: Dict, prop_names: List) -> Callable:
        """
        Get the transform function for all the features.
        """
        prop_pos_index = {prop_name: i for i,
                          prop_name in enumerate(prop_names)}

        def transform_function(prop_values):
            ret_feature = []
            for feature_name in features:
                feature = features[feature_name]
                feature_props = feature.get('prop')

                feature_prop_names = [prop.get('name')
                                      for prop in feature_props]
                feature_prop_types = [prop.get('type')
                                      for prop in feature_props]
                feature_prop_values = []
                for index, prop_name in enumerate(feature_prop_names):
                    raw_value = prop_values[prop_pos_index[prop_name]]
                    # convert byte value according to type
                    feature_prop_values.append(
                        getattr(
                            raw_value, NEBULA_TYPE_MAP[feature_prop_types[index]])()
                    )
                if feature.get('type') == 'value':
                    ret_feature.append(feature_prop_values[0])
                elif feature.get('type') == 'enumeration':
                    enumeration_dict = feature.get('enumeration')
                    ret_feature.append(enumeration_dict.get(
                        feature_prop_values[0], ENUMERATION_DEFAULT_VALUE))
                elif feature.get('type') == 'function':
                    feature_filter_function = feature.get('function')
                    ret_feature.append(
                        feature_filter_function(*feature_prop_values))
            if len(ret_feature) == 0:
                return None
            else:
                return ret_feature

        return transform_function

    def load_vertexes(self) -> Tuple[Dict, Dict]:
        """
        Load the part vertexes data from Nebula Graph Cluster, return vertex ids and ndata.
        """
        # generate vertices per tag
        vertex_id_dict = dict()
        ndata = dict()

        # assumed only one graph space though, we iterate it here.
        for space_name in self.tag_feature_dict:
            if space_name not in vertex_id_dict:
                vertex_id_dict[space_name] = dict()
            for tag_name in self.tag_feature_dict[space_name]:
                if tag_name not in vertex_id_dict[space_name]:
                    vertex_id_dict[space_name][tag_name] = dict()
                _vertex_id_dict = vertex_id_dict[space_name][tag_name]
                tag_features = self.tag_feature_dict[space_name][tag_name]
                props = set()
                for feature_name in tag_features:
                    feature = tag_features[feature_name]
                    if feature_name not in ndata:
                        ndata[feature_name] = {tag_name: []}
                    else:
                        assert tag_name not in ndata[feature_name], \
                            f'tag {tag_name} already exists in ndata[{feature_name}]'
                        ndata[feature_name][tag_name] = []
                    for prop in feature.get('prop', []):
                        props.add(prop['name'])
                prop_names = list(props)

                graph_storage_client = self.get_storage_client()
                resp = graph_storage_client.scan_vertex_with_part(
                    space_name=space_name,
                    part=self.part_id,
                    tag_name=tag_name,
                    prop_names=prop_names)
                vertex_index = 0
                transform_function = self.get_feature_transform_function(
                    tag_features, prop_names)
                while resp.has_next():
                    result = resp.next()
                    for vertex_data in result:
                        _vertex_id_dict[vertex_data.get_id()] = vertex_index
                        vertex_index += 1
                        # feature data for vertex(node)
                        if not tag_features:
                            continue
                        prop_values = vertex_data.get_prop_values()
                        feature_values = transform_function(prop_values)
                        for index, feature_name in enumerate(tag_features):
                            feature = tag_features[feature_name]
                            if feature_name not in ndata:
                                ndata[feature_name] = {tag_name: []}
                            ndata[feature_name][tag_name].append(
                                feature_values[index])
                if prop_names:
                    assert vertex_index == len(
                        ndata[feature_name][tag_name]), \
                        f'vertex index {vertex_index} != len(ndata[{prop_names[0]}][{tag_name}])'

        return vertex_id_dict, ndata

    def load_edges(self, vertex_id_dict) -> Tuple[Dict, Dict]:
        """
        Load the part edge data from Nebula Graph Cluster, return edge and edata
        """
        data_dict = dict()
        edata = dict()
        for space_name in self.edge_feature_dict:

            for edge_name in self.edge_feature_dict[space_name]:
                edge = self.edge_feature_dict[space_name][edge_name]
                edge_features = edge.get('features', {})
                start_vertex_tag, end_vertex_tag = edge.get(
                    'start_vertex_tag'), edge.get('end_vertex_tag')
                assert (start_vertex_tag, edge_name, end_vertex_tag) not in data_dict, \
                    f'edge {start_vertex_tag}-{edge_name}-{end_vertex_tag} already exists'
                props = set()
                for feature_name in edge_features:
                    feature = edge_features[feature_name]
                    if feature_name not in edata:
                        edata[feature_name] = {edge_name: []}
                    else:
                        assert edge_name not in edata[feature_name], \
                            f'tag {edge_name} already exists in edata[{feature_name}]'
                        edata[feature_name][edge_name] = []
                    for prop in feature.get('prop', []):
                        props.add(prop['name'])
                prop_names = list(props)

                graph_storage_client = self.get_storage_client()
                resp = graph_storage_client.scan_edge_with_part(
                    space_name=space_name,
                    part=self.part_id,
                    edge_name=edge_name,
                    prop_names=prop_names)
                transform_function = self.get_feature_transform_function(
                    edge_features, prop_names)
                start_vertices, end_vertices = [], []
                start_vertex_id_dict = vertex_id_dict[space_name][start_vertex_tag]
                end_vertex_id_dict = vertex_id_dict[space_name][end_vertex_tag]
                while resp.has_next():
                    result = resp.next()
                    for edge_data in result:
                        # edge data for edge
                        start_vertices.append(
                            start_vertex_id_dict[edge_data.get_src_id()]
                        )
                        end_vertices.append(
                            end_vertex_id_dict[edge_data.get_dst_id()]
                        )

                        # feature data for edge
                        if not edge_features:
                            continue
                        prop_values = edge_data.get_prop_values()
                        feature_values = transform_function(prop_values)
                        for index, feature_name in enumerate(edge_features):
                            feature = edge_features[feature_name]
                            if feature_name not in edata:
                                edata[feature_name] = {edge_name: []}
                            edata[feature_name][edge_name].append(
                                feature_values[index])

                data_dict[(start_vertex_tag, edge_name, end_vertex_tag)] = (
                    start_vertices, end_vertices)

        return data_dict, edata
