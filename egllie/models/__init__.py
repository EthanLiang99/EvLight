from egllie.models.egretinex import EgLlie

def get_model(config):
    if config.NAME == "egretinex":
        return EgLlie(
            config
        )
    else:
        raise ValueError(f"Model {config.NAME} is not supported.")
