from flask import jsonify, session
from functools import wraps


def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "role" not in session:
                return jsonify({"error": "Unauthorized"}), 401
            if session["role"] not in allowed_roles:
                return jsonify({"error": "Access denied"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator