from tycho.feature.category import CategoricalEmbedder
from tycho.feature.number import NumericNormalizer


def detect_column_type(data):
    try:
        data_as_float = [float(x) for x in data]
        if len(data_as_float) == sum([x.is_integer() for x in data_as_float]):
            return 'int'
        else:
            return 'float'
    except:
        return 'category'

def detect_df_types(df):
    type_dictionary = {}
    for col in df.columns:
        type_dictionary[col] = detect_column_type(df[col])
    return type_dictionary

def get_featurizers_class(type_dictionary):
    featurizer_class_dict = {}
    for col, dtype in type_dictionary.items():
        if dtype in ['float', 'int']:
            featurizer_class_dict[col] = NumericNormalizer
        else:
            featurizer_class_dict[col] = CategoricalEmbedder
    return featurizer_class_dict
