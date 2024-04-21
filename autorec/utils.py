import yaml


def load_config(path: str) -> dict:
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    return config

test = load_config("./config.yml")