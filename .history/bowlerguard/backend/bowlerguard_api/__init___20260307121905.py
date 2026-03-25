from flask import Flask
from .extensions import cors
from .routes.auth_routes import auth_bp
from .routes.core_routes import core_bp
from .routes.prediction_routes import prediction_bp


def create_app():
    app = Flask(__name__)
    app.secret_key = "bowlerguard_secret"

    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = False

    cors.init_app(
        app,
        supports_credentials=True,
        resources={
            r"/*": {
                "origins": [
                    "http://127.0.0.1:5500",
                    "http://localhost:5500"
                ]
            }
        }
    )

    app.register_blueprint(core_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)

    return app