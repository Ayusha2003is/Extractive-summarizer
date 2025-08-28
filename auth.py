from flask import Blueprint, request, jsonify
from models import db, User
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token
import datetime
import traceback

auth = Blueprint("auth", __name__)
bcrypt = Bcrypt()

def set_bcrypt_instance(bcrypt_instance):
    global bcrypt
    bcrypt = bcrypt_instance

@auth.route('/signup', methods=['POST'])
def signup():
    try:
        print("Signup route called")  # Debug log
        data = request.get_json()
        print(f"Received data: {data}")  # Debug log
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")
        
        if not username:
            return jsonify({"error": "Username is required"}), 400
        if not email:
            return jsonify({"error": "Email is required"}), 400
        if not password:
            return jsonify({"error": "Password is required"}), 400

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({"error": "Email already exists"}), 409

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_pw)
        
        db.session.add(new_user)
        db.session.commit()
        
        print(f"User created successfully: {username}")  # Debug log
        return jsonify({"message": "User created successfully"}), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"Signup error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@auth.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        print("Handling OPTIONS request for /auth/login")  # Debug log
        return '', 200  # Respond to preflight request

    try:
        print("Login route called")  # Debug log
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            access_token = create_access_token(
                identity={"email": user.email, "username": user.username},
                expires_delta=datetime.timedelta(days=1)
            )
            return jsonify({
                "message": "Login successful",
                "access_token": access_token,
                "user": {
                    "username": user.username,
                    "email": user.email
                }
            }), 200
        else:
            return jsonify({"error": "Invalid email or password"}), 401
            
    except Exception as e:
        print(f"Login error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500