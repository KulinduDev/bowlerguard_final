from flask import Blueprint, jsonify, request, session
from ..auth_store import USERS

auth_bp = Blueprint("auth", __name__)


@auth_bp.get("/session-status")
def session_status():
    if "user" in session and "role" in session:
        return jsonify({
            "logged_in": True,
            "username": session["user"],
            "role": session["role"]
        })
    return jsonify({"logged_in": False})


@auth_bp.post("/login")
def login():
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get("password")

    user = USERS.get(username)
    if user and user["password"] == password:
        session["user"] = username
        session["role"] = user["role"]
        return jsonify({
            "message": "Login successful",
            "username": username,
            "role": user["role"]
        })

    return jsonify({"error": "Invalid credentials"}), 401


@auth_bp.post("/logout")
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})