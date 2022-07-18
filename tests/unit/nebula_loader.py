import unittest

from nebula_dgl import NebulaLoader

from nebula3.common import ttypes
from nebula3.data.DataObject import ValueWrapper

# Test NebulaLoader
class TestNebulaLoader(unittest.TestCase):
    def setUp(self):
        self.nebula_config = {
            "graph_hosts": [
                        ('graphd', 9669),
                        ('graphd1', 9669),
                        ('graphd2', 9669)
                    ],
            "user": "root",
            "password": "nebula",
        }
        with open('example/nebula_to_dgl_mapper.yaml', 'r') as f:
            self.feature_mapper = yaml.safe_load(f)
        
        # Load the NebulaLoader
        self.nebula_loader = NebulaLoader(self.nebula_config, self.feature_mapper)
        self.assertIsNotNone(self.nebula_loader)
    
    def test_get_feature_transform_function(self):
        tag_features = self.nebula_loader.tag_feature_dict[
            "basketballplayer"]["player"]
        props = set()
        for feature_name in tag_features:
            feature = tag_features[feature_name]
            for prop in feature.get('prop', []):
                props.add(prop['name'])
        prop_names = list(props)
        transform_function = self.nebula_loader.get_feature_transform_function(
            tag_features, prop_names)
        self.assertIsNotNone(transform_function)
        value = ttypes.Value()
        value.set_iVal(30)
        value_wrapper = ValueWrapper(value)
        feature_value = transform_function([value_wrapper])
        self.assertEqual(feature_value[0], 30 / 100)

    def test_load(self):
        dgl_graph = self.nebula_loader.load()
        self.assertIsNotNone(dgl_graph)

unittest.main()