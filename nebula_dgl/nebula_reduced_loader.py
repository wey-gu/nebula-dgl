from nebula3.mclient import MetaCache, MetaClient
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config as NebulaConfig
from nebula3.common import ttypes

from dgl import DGLHeteroGraph, heterograph
from torch import tensor
from concurrent.futures import ThreadPoolExecutor

from typing import Dict

import logging

from nebula_dgl.nebula_part_loader import NebulaPartLoader

VALID_FILTERS = ['function', 'enumeration', 'value']
logger = logging.getLogger(__package__)


class NebulaReducedLoader():
    """
    NebulaReducedLoader is a class that reduces vertexes and edges data from Nebula Graph as a DGLGraph.

    feature_mapper: a dictionary that maps the feature names to the feature,
                    its equivalent example in YAML is:
                    ../example/nebula_to_dgl_mapper.yaml

    """

    def __init__(self, nebula_config: Dict, feature_mapper: Dict):
        """
        Initialize the NebulaLoader class.
        """
        self.nebula_config = nebula_config
        self.feature_mapper = feature_mapper
        self.part_nums = None
        self.tag_feature_dict = None
        self.edge_feature_dict = None
        self.meta_hosts = None
        self.init_connection()
        self.remap_vertex_id = feature_mapper.get('remap_vertex_id', True)
        if self.remap_vertex_id:
            self.vertex_id_dict = dict()
        self.validate_feature_mapper(feature_mapper)

    def init_connection(self):
        """
        Get the connection to the Nebula Graph.
        """
        user = self.nebula_config.get('nebula_user', 'root')
        password = self.nebula_config.get('nebula_password', 'nebula')
        graph_hosts = self.nebula_config.get(
            'graph_hosts',
            [
                ('graphd', 9669),
                ('graphd1', 9669),
                ('graphd2', 9669)
            ]
        )
        config = NebulaConfig()
        config.max_connection_pool_size = 10
        self.connection_pool = ConnectionPool()
        self.connection_pool.init(graph_hosts, config)
        # get meta_hosts
        with self.connection_pool.session_context(user, password) as session:
            meta_hosts = []
            result = session.execute('SHOW HOSTS META')
            assert result.is_succeeded() and result.error_code() == 0
            assert result.row_size() > 0
            for row in result.rows():
                meta_hosts.append(
                    (row.values[0].get_sVal().decode(),
                     row.values[1].get_iVal())
                )
        self.meta_hosts = meta_hosts

        # get part_nums
        space_name = self.feature_mapper.get('space', None)
        with self.connection_pool.session_context(user, password) as session:
            result = session.execute('USE {}; SHOW PARTS'.format(space_name))
            assert result.is_succeeded() and result.error_code() == 0
            self.part_nums = len(result.rows())

    def get_meta_client(self):
        """
        Get the MetaClient.
        """
        meta_client = MetaClient(self.meta_hosts, 50000)
        meta_client.open()
        return meta_client

    def validate_feature_mapper(self, feature_mapper: Dict):
        """
        Validate the feature mapper, parse the feature_mapper.yaml file.

        feature_mapper: a dictionary that maps the feature names to the feature
        """
        # ensure all properties exist in nebula graph schema
        m_client = self.get_meta_client()
        self.spaces_dict = {
            space.name.decode(): space.id.get_space_id()
            for space in m_client.list_spaces()}
        self.vertex_tag_schema_dict = {}
        self.tag_feature_dict = {}
        self._validate_vertex_tags(m_client, feature_mapper)
        self.edge_type_schema_dict = {}
        self.edge_feature_dict = {}
        self._validate_edge_types(m_client, feature_mapper)

    def _validate_vertex_tags(self, m_client: MetaClient, feature_mapper: Dict):
        """
        Validate the vertex tag.

        TODO: add validation for remap_vertex_id.
        """
        space_name = feature_mapper.get('space', None)
        self.vertex_id_type = feature_mapper.get('vertex_id_type', 'int')
        assert space_name is not None, 'space is required in mapper'
        for vertex_tag in feature_mapper.get('vertex_tags', []):
            # ensure space exists in Nebula Graph
            assert space_name in self.spaces_dict, \
                'space {} does not exist'.format(space_name)
            tag_name = vertex_tag.get('name')

            if space_name not in self.vertex_tag_schema_dict:
                self.vertex_tag_schema_dict[space_name] = {
                    tag.tag_name.decode(): tag for tag in m_client.list_tags(
                        self.spaces_dict[space_name])
                }

            # ensure tag exists
            assert tag_name in self.vertex_tag_schema_dict[space_name], \
                'tag {} does not exist'.format(tag_name)
            if space_name not in self.tag_feature_dict:
                self.tag_feature_dict[space_name] = {tag_name: {}}
            if tag_name not in self.tag_feature_dict[space_name]:
                self.tag_feature_dict[space_name][tag_name] = {}
            for feature in vertex_tag.get('features', []):
                feature_name = feature.get('name')
                assert feature_name is not None, \
                    'feature name is not specified in {}'.format(feature)
                # ensure properties exists and type is correct
                tag = self.vertex_tag_schema_dict[space_name][tag_name]
                tag_props_types = {
                    prop.name.decode(): ttypes.PropertyType._VALUES_TO_NAMES[
                        prop.type.type] for prop in tag.schema.columns}
                for prop in feature.get('properties', []):
                    assert prop.get('name') in tag_props_types, \
                        'property {} does not exist in {}'.format(
                            prop.get('name'), tag_name)

                    assert prop.get('type').upper(
                    ) in tag_props_types[prop.get('name')], \
                        'property {} is not of type {} in {}'.format(
                            prop.get('name'), prop.get('type'), tag_name)

                # ensure filter exists
                filter = feature.get('filter')
                assert filter is not None
                assert filter.get('type') in VALID_FILTERS
                if filter.get('type') == 'function':
                    function = filter.get('function')
                    assert function is not None, \
                        'function is missing in filter: { }'.format(filter)
                    try:
                        self.tag_feature_dict[space_name][tag_name][feature_name] = {
                            'type': 'function',
                            'function': eval(function),
                            'prop': feature.get('properties')}
                    except Exception as e:
                        # function is not evaluable
                        logger.error(
                            'Function {} is not evaluable: {}'.format(
                                function, e))
                        raise e
                if filter.get('type') == 'enumeration':
                    enumeration = filter.get('enumeration')
                    assert enumeration is not None, 'enumeration is None'
                    assert isinstance(
                        enumeration, dict), 'enumeration is not a dict'
                    assert len(enumeration) > 0, 'enumeration is empty'
                    self.tag_feature_dict[space_name][tag_name][feature_name] = {
                        'type': 'enumeration',
                        'enumeration': enumeration,
                        'prop': feature.get('properties')}
                if filter.get('type') == 'value':
                    # value is not used in this case
                    assert len(feature.get('properties', [])) == 1, \
                        'value filter can only have one property'
                    self.tag_feature_dict[space_name][tag_name][feature_name] = {
                        'type': 'value',
                        'prop': feature.get('properties')}

        assert len(self.tag_feature_dict) == 1, \
            f'There should be only 1 graph space involved, '\
            f'but now: {self.tag_feature_dict.keys()}'

    def _validate_edge_types(self, m_client: MetaClient, feature_mapper: Dict):
        """
        Validate the edge type.
        """
        space_name = feature_mapper.get('space', None)
        assert space_name is not None, 'space is required in mapper'
        for edge_type in feature_mapper.get('edge_types', []):
            # ensure space exists in Nebula Graph
            assert space_name in self.spaces_dict, \
                'space {} does not exist'.format(space_name)
            if space_name not in self.edge_type_schema_dict:
                self.edge_type_schema_dict[space_name] = {
                    edge.edge_name.decode(): edge for edge in m_client.list_edges(
                        self.spaces_dict[space_name])
                }

            # ensure edge exists
            edge_name = edge_type.get('name')
            assert edge_name in self.edge_type_schema_dict[space_name], \
                'edge {} does not exist'.format(edge_name)
            if space_name not in self.edge_feature_dict:
                self.edge_feature_dict[space_name] = {
                    edge_name: {
                        "start_vertex_tag": edge_type.get('start_vertex_tag'),
                        "end_vertex_tag": edge_type.get('end_vertex_tag'),
                        "features": {}
                    }}
            if edge_name not in self.edge_feature_dict[space_name]:
                self.edge_feature_dict[space_name][edge_name] = {
                    "start_vertex_tag": edge_type.get('start_vertex_tag'),
                    "end_vertex_tag": edge_type.get('end_vertex_tag'),
                    "features": {}
                }
            for feature in edge_type.get('features', []):
                feature_name = feature.get('name')
                assert feature_name is not None, \
                    'feature name is not specified in {}'.format(feature)
                # ensure properties exists and type is correct
                edge = self.edge_type_schema_dict[space_name][edge_name]
                edge_props_types = {
                    prop.name.decode(): ttypes.PropertyType._VALUES_TO_NAMES[
                        prop.type.type] for prop in edge.schema.columns}
                for prop in feature.get('properties', []):
                    assert prop.get('name') in edge_props_types, \
                        'property {} does not exist in {}'.format(
                            prop.get('name'), edge_name)
                    assert prop.get('type').upper(
                    ) in edge_props_types[prop.get('name')], \
                        'property {} is not of type {} in {}'.format(
                            prop.get('name'), prop.get('type'), edge.edge_name.decode())
                # ensure filter exists
                filter = feature.get('filter')
                assert filter is not None
                assert filter.get('type') in VALID_FILTERS
                if filter.get('type') == 'function':
                    function = filter.get('function')
                    assert function is not None, \
                        'function is missing in filter: { }'.format(filter)
                    try:
                        self.edge_feature_dict[space_name][edge_name]['features'][feature_name] = {
                            'type': 'function',
                            'function': eval(function),
                            'prop': feature.get('properties')}
                    except Exception as e:
                        # function is not evaluable
                        logger.error(
                            'Function {} is not evaluable: {}'.format(
                                function, e))
                        raise e
                if filter.get('type') == 'enumeration':
                    enumeration = filter.get('enumeration')
                    assert enumeration is not None, 'enumeration is None'
                    assert isinstance(
                        enumeration, dict), 'enumeration is not a dict'
                    assert len(enumeration) > 0, 'enumeration is empty'
                    self.edge_feature_dict[space_name][edge_name]['features'][feature_name] = {
                        'type': 'enumeration',
                        'enumeration': enumeration,
                        'prop': feature.get('properties')}
                if filter.get('type') == 'value':
                    # value is not used in this case
                    assert len(feature.get('properties', [])) == 1, \
                        'value filter can only have one property'
                    self.edge_feature_dict[space_name][edge_name]['features'][feature_name] = {
                        'type': 'value',
                        'prop': feature.get('properties')}

        assert len(self.edge_type_schema_dict) == 1, \
            f'There should be only 1 graph space involved, '\
            f'but now: {self.edge_type_schema_dict.keys()}'
        assert self.edge_type_schema_dict.keys() == self.edge_feature_dict.keys(), \
            f'edge type schema and feature dict should have the same graph space, '\
            f'but now: {self.edge_type_schema_dict.keys()} and {self.edge_feature_dict.keys()}'

    def load(self) -> DGLHeteroGraph:
        # load vertexes
        part_loaders = []
        for i in range(1, self.part_nums + 1):
            part = NebulaPartLoader(
                i, self.meta_hosts, self.edge_feature_dict, self.tag_feature_dict)
            part_loaders.append(part)

        vertexes_futures = []

        with ThreadPoolExecutor(max_workers=self.part_nums) as executor:
            for p in part_loaders:
                fu = executor.submit(p.load_vertexes)
                vertexes_futures.append(fu)

        vertex_ids = vertexes_futures[0].result()[0]
        ndata = vertexes_futures[0].result()[1]

        for p in vertexes_futures[1:]:
            vertexes = p.result()[0]
            for space, tag_vertexes in vertexes.items():
                for tag, ids in tag_vertexes.items():
                    vid_index = len(vertex_ids[space][tag])
                    for vid, _ in ids.items():
                        vertex_ids[space][tag][vid] = vid_index
                        vid_index += 1
            vertexes_pro = p.result()[1]
            for prop, tag_vertexes in vertexes_pro.items():
                for tag, ids in tag_vertexes.items():
                    ndata[prop][tag].extend(ids)

        # load edges
        edges_futures = []
        with ThreadPoolExecutor(max_workers=self.part_nums) as executor:
            for p in part_loaders:
                fu = executor.submit(p.load_edges, vertex_ids)
                edges_futures.append(fu)

        data_dict = edges_futures[0].result()[0]
        edata = edges_futures[0].result()[1]
        for f in edges_futures[1:]:
            edges = f.result()[0]
            for key, value in edges.items():
                data_dict[key][0].extend(value[0])
                data_dict[key][1].extend(value[1])
            edges_pro = f.result()[1]
            for prop, edge_prop in edges_pro.items():
                for key, value in edge_prop.items():
                    edata[prop][key].extend(value)

        for edge, edge_data in data_dict.items():
            data_dict[edge] = (tensor(edge_data[0]), tensor(edge_data[1]))

        # Convert to DGLHeteroGraph
        space_name = self.feature_mapper.get('space', None)
        dgl_graph: DGLHeteroGraph = heterograph(data_dict)
        for prop_name, tag_dict in ndata.items():
            for tag_name, prop_values in tag_dict.items():
                dgl_graph.ndata[prop_name] = tensor(prop_values) if len(self.tag_feature_dict[space_name]) == 1 else \
                    {tag_name: tensor(prop_values)}
        for prop_name, edge_dict in edata.items():
            for edge_name, prop_values in edge_dict.items():
                dgl_graph.edata[prop_name] = tensor(prop_values) if len(self.edge_feature_dict[space_name]) == 1 else \
                    {edge_name: tensor(prop_values)}

        return dgl_graph
