from flask import Blueprint, jsonify, current_app

core_bp = Blueprint("core", __name__)


@core_bp.get("/")
def index():
    return current_app.send_static_file("index.html")


@core_bp.get("/health")
def health():
    return jsonify({"status": "ok"})