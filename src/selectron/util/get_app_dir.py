from pathlib import Path

from platformdirs import user_config_dir

APP_DIR_NAME = "selectron"


def get_app_dir() -> Path:
    app_dir = Path(user_config_dir(APP_DIR_NAME))
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir
