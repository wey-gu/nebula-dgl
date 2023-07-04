from nebula3.mclient import MetaCache, MetaClient
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config as NebulaConfig
from nebula3.common import ttypes
from nebula3.data.DataObject import Node, Relationship, PathWrapper

import pandas as pd
from nebula3.data.DataObject import Value, ValueWrapper
from nebula3.data.ResultSet import ResultSet

from dgl import DGLHeteroGraph, heterograph
from torch import tensor

from typing import Dict, List, Callable

import logging


VALID_FILTERS = ['function', 'enumeration', 'value']
logger = logging.getLogger(__package__)
ENUMERATION_DEFAULT_VALUE = -1
NEBULA_TYPE_MAP = {
    "int": "as_int",
    "double": "as_double",
    "str": "as_string",
    "float": "as_double",
}

SCAN_MODE = "scan_mode"
QUERY_MODE = "query_mode"



# NOTE: this was to add back capability to Nebula-Python 3.2.0

cast_as = {
    Value.NVAL: "as_null",
    Value.BVAL: "as_bool",
    Value.IVAL: "as_int",
    Value.FVAL: "as_double",
    Value.SVAL: "as_string",
    Value.LVAL: "as_list",
    Value.UVAL: "as_set",
    Value.MVAL: "as_map",
    Value.TVAL: "as_time",
    Value.DVAL: "as_date",
    Value.DTVAL: "as_datetime",
    Value.VVAL: "as_node",
    Value.EVAL: "as_relationship",
    Value.PVAL: "as_path",
    Value.GGVAL: "as_geography",
    Value.DUVAL: "as_duration",
}

def result_to_df(result: ResultSet) -> pd.DataFrame:
    """
    build list for each column, and transform to dataframe
    """
    columns = result.keys()
    d: Dict[str, list] = {}
    for col_num in range(result.col_size()):
        col_name = columns[col_num]
        col_list = result.column_values(col_name)
        d[col_name] = [cast(x) for x in col_list]
    return pd.DataFrame(d)


def cast(val: ValueWrapper):
    _type = val._value.getType()
    if _type == Value.__EMPTY__:
        return None
    if _type in cast_as:
        return getattr(val, cast_as[_type])()
    if _type == Value.LVAL:
        return [x.cast() for x in val.as_list()]
    if _type == Value.UVAL:
        return {x.cast() for x in val.as_set()}
    if _type == Value.MVAL:
        return {k: v.cast() for k, v in val.as_map().items()}


class NebulaLoader():
    """
    NebulaLoader is a class that loads the Nebula Graph as a DGLGraph.

    feature_mapper: a dictionary that maps the feature names to the feature,
                    its equivalent example in YAML is:
                    ../example/nebula_to_dgl_mapper.yaml

    """

    def __init__(self, nebula_config: Dict, feature_mapper: Dict, query: str = None, query_space: str = None):
        """
        Initialize the NebulaLoader class.
        """
        self.nebula_config = nebula_config
        self.init_connection()
        if query is None:
            self.mode = SCAN_MODE

        else:
            self.mode = QUERY_MODE
            assert query_space is not None, "query_space must be specified in query mode"
            self.query = query
            self.query_space = query_space

        self.remap_vertex_id = feature_mapper.get('remap_vertex_id', True)
        if self.remap_vertex_id:
            self.vertex_id_dict = dict()
        self.validate_feature_mapper(feature_mapper)

    def init_connection(self):
        """
        Get the connection to the Nebula Graph.
        """
        self._user = self.nebula_config.get('nebula_user', 'root')
        self._password = self.nebula_config.get('nebula_password', 'nebula')
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
        with self.connection_pool.session_context(self._user, self._password) as session:
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

    def get_meta_client(self):
        """
        Get the MetaClient.
        """
        meta_client = MetaClient(self.meta_hosts, 50000)
        meta_client.open()
        return meta_client

    def get_storage_client(self):
        """
        Get the GraphStorageClient.
        """
        meta_cache = MetaCache(self.meta_hosts, 50000)
        storage_client = GraphStorageClient(meta_cache)
        return storage_client

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
        self.prop_pos_index = {}
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
            # build self.prop_pos_index
            if tag_name not in self.prop_pos_index:
                self.prop_pos_index[tag_name] = dict()
            tag = self.vertex_tag_schema_dict[space_name][tag_name]
            for index, prop in enumerate(tag.schema.columns):
                self.prop_pos_index[tag_name][prop.name.decode()] = index

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

            # build self.prop_pos_index
            edge_name = edge_type.get('name')
            if edge_name not in self.prop_pos_index:
                self.prop_pos_index[edge_name] = dict()
            edge = self.edge_type_schema_dict[space_name][edge_name]
            for index, prop in enumerate(edge.schema.columns):
                self.prop_pos_index[edge_name][prop.name.decode()] = index

            # ensure edge exists

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
                    #raw_value = prop_values[self.prop_pos_index[tag_or_edge][prop_name]]
                    raw_value = prop_values[prop_pos_index[prop_name]]
                    # convert byte value according to type
                    feature_prop_values.append(
                        getattr(raw_value, NEBULA_TYPE_MAP[feature_prop_types[index]])()
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

    def _load_in_scan_mode(self) -> DGLHeteroGraph:
        """
        Load the graph in scan mode.
        """

        data_dict = dict()

        vertex_id_dict = dict()
        ndata = dict()
        edata = dict()

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
                resp = graph_storage_client.scan_vertex(
                    space_name=space_name,
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
                            ndata[feature_name][tag_name].append(feature_values[index])
                if prop_names:
                    assert vertex_index == len(
                        ndata[feature_name][tag_name]), \
                        f'vertex index {vertex_index} != len(ndata[{prop_names[0]}][{tag_name}])'

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
                resp = graph_storage_client.scan_edge(
                    space_name=space_name,
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
                            edata[feature_name][edge_name].append(feature_values[index])

                data_dict[(start_vertex_tag, edge_name, end_vertex_tag)] = (
                    tensor(start_vertices), tensor(end_vertices))
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

    def parse_result(self, g: Dict, item):
        if isinstance(item, Node):
            node_id = item.get_id().cast()
            tags = item.tags()  # list of strings
            for tag in tags:
                props = item.properties(tag)
                g["nodes"][tag][node_id] = props
        elif isinstance(item, Relationship):
            src_id = item.start_vertex_id().cast()
            dst_id = item.end_vertex_id().cast()
            edge_name = item.edge_name()
            props = item.properties()
            # NOTE: we didn't handle rank here for now
            g["edges"][edge_name][(src_id, dst_id)] = props

        elif isinstance(item, PathWrapper):
            for node in item.nodes():
                self.parse_result(g, node)
            for edge in item.relationships():
                self.parse_result(g, edge)
        elif isinstance(item, list):
            for it in item:
                self.parse_result(g, it)

    def _load_in_query_mode(self) -> DGLHeteroGraph:
        """
        load in query mode
        """
        g: Dict = {
            "space_name": self.query_space,
            "edges": {edge_type: {} for edge_type in self.edge_type_schema_dict[self.query_space]},
            "nodes": {tag: {} for tag in self.vertex_tag_schema_dict[self.query_space]}
        }

        with self.connection_pool.session_context(self._user, self._password) as session:
            session.execute(f'USE {self.query_space}')
            result: ResultSet = session.execute(self.query)

            assert (result.is_succeeded() and result.error_code() == 0), (
                f"failed to query: {self.query} on space: {self.query_space}\n"
                f"error: {result.error_msg}"
            )

        result_df = result_to_df(result)
        for _, row in result_df.iterrows():
            for item in row:
                self.parse_result(g, item)

        data_dict = dict()

        vertex_id_dict = dict()
        ndata = dict()
        edata = dict()

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

                vertex_index = 0
                transform_function = self.get_feature_transform_function(
                    tag_features, prop_names)
                for vertex_id, prop_map in g['nodes'][tag_name].items():
                    _vertex_id_dict[vertex_id] = vertex_index
                    vertex_index += 1
                    # feature data for vertex(node)
                    if not tag_features:
                        continue
                    prop_values = [prop_map.get(prop_name) for prop_name in prop_names]
                    feature_values = transform_function(prop_values)
                    for index, feature_name in enumerate(tag_features):
                        feature = tag_features[feature_name]
                        if feature_name not in ndata:
                            ndata[feature_name] = {tag_name: []}
                        ndata[feature_name][tag_name].append(feature_values[index])

                if prop_names:
                    assert vertex_index == len(
                        ndata[feature_name][tag_name]), \
                        f'vertex index {vertex_index} != len(ndata[{prop_names[0]}][{tag_name}])'

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

                transform_function = self.get_feature_transform_function(
                    edge_features, prop_names)
                start_vertices, end_vertices = [], []
                start_vertex_id_dict = vertex_id_dict[space_name][start_vertex_tag]
                end_vertex_id_dict = vertex_id_dict[space_name][end_vertex_tag]

                for edge_tuple, prop_map in g['edges'][edge_name].items():
                    start_vertices.append(
                        start_vertex_id_dict[edge_tuple[0]]
                    )
                    end_vertices.append(
                        end_vertex_id_dict[edge_tuple[1]]
                    )
                    # feature data for edge
                    if not edge_features:
                        continue
                    prop_values = [prop_map.get(prop_name) for prop_name in prop_names]
                    feature_values = transform_function(prop_values)
                    for index, feature_name in enumerate(edge_features):
                        feature = edge_features[feature_name]
                        if feature_name not in edata:
                            edata[feature_name] = {edge_name: []}
                        edata[feature_name][edge_name].append(feature_values[index])

                data_dict[(start_vertex_tag, edge_name, end_vertex_tag)] = (
                    tensor(start_vertices), tensor(end_vertices))
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

    def load(self) -> DGLHeteroGraph:
        """
        Load the data from Nebula Graph Cluster, return a DGL heterograph.
        ref: https://github.com/dmlc/dgl/blob/master/python/dgl/convert.py::heterograph
        """
        if self.mode == SCAN_MODE:
            return self._load_in_scan_mode()
        elif self.mode == QUERY_MODE:
            return self._load_in_query_mode()
        else:
            raise ValueError(f'unknown mode {self.mode}')
