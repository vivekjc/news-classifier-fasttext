import unicodedata
import re
import yaml
from datasets import load_dataset

def normalize_and_clean_text(raw_text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKD', raw_text).replace('\xa0', ' ')).strip()

def fetch_datasets_from_config(config_file_path="datasets_config.yaml"):
    with open(config_file_path, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)

    loaded_datasets = {}
    for dataset_key, dataset_info in config_data['datasets'].items():
        loaded_datasets[dataset_key] = load_dataset(
            dataset_info['source'],
            name=dataset_info['name'],
            split=dataset_info['split'],
            streaming=dataset_info.get('streaming', False)
        )
    return loaded_datasets