from flask import Flask
from .extensions import cors
from .routes.auth_routes import auth_bp
from .routes.core_routes import core_bp
from .routes.prediction_routes import prediction_bp


def create_app():
    app = Flask(__name__)
    app.secret_key = "bowlerguard_secret"

    cors.init_app(app, supports_credentials=True)

    app.register_blueprint(core_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)

    return app