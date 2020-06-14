import yaml

config_file = open('cv_config.yaml', 'r')
config = yaml.load(config_file, Loader=yaml.FullLoader)
