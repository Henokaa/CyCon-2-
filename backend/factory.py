from flask import Flask, g
from logging.config import dictConfig

from backend import auth, home, config, experiments
from backend.db import DBManager

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {"class": "logging.StreamHandler", "formatter": "default"}
        },
        "root": {"level": "DEBUG", "handlers": ["wsgi"]},
    }
)


def create_app(password_file="/run/secrets/db-password"):
    app = Flask(__name__, static_folder="static")
    app.config.from_mapping(**config.config)
    app.config["DB_PASSWORD"] = password_file

    # security
    app.config.update(
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )

    app.register_blueprint(auth.bp)
    app.register_blueprint(home.bp)
    app.register_blueprint(experiments.bp)

    app.add_url_rule("/", endpoint="index")

    return app
