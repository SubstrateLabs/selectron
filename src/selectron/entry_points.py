from selectron.cli.app import SelectronApp
from selectron.util.model_config import ModelConfig


def start():
    model_config = ModelConfig()

    app = SelectronApp(model_config=model_config)
    app.run()


if __name__ == "__main__":
    start()
