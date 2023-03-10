"""Flask App"""
import sys

from flask import Flask
from flask_basicauth import BasicAuth
from flask_injector import FlaskInjector

from autotim.app.dependencies import configure
from autotim.app.authconfig import AuthConfig

from autotim.app.endpoints.store_bp import STORE_BP
from autotim.app.endpoints.predict_bp import PREDICT_BP
from autotim.app.endpoints.train_bp import TRAIN_BP


# when run in docker, recursion limit is capped at 1000
sys.setrecursionlimit(3000)


def register_blueprints(application):
    application.register_blueprint(STORE_BP)
    application.register_blueprint(TRAIN_BP)
    application.register_blueprint(PREDICT_BP)


def create_app():
    application = Flask(__name__)
    application.config['JSON_SORT_KEYS'] = False

    application.config["BASIC_AUTH_FORCE"] = True
    application.config["BASIC_AUTH_USERNAME"] = AuthConfig.AUTOTIM_USERNAME
    application.config["BASIC_AUTH_PASSWORD"] = AuthConfig.AUTOTIM_PASSWORD
    basic_auth = BasicAuth(application)

    register_blueprints(application)
    return application, basic_auth


app, auth = create_app()
FlaskInjector(app=app, modules=[configure])


@app.route("/")
def hello():
    return "Hello, World!"


if __name__ == "__main__":
    app.run(port=5004, debug=True)
