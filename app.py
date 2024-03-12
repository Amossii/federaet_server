from flask import Flask,session,g,request,jsonify
from exts import db
import config
from models import User
from blueprints.file import bp as file_bp
from blueprints.auth import bp as auth_bp
from blueprints.test import bp as test_bp
from blueprints.resources import bp as resources_bp
from blueprints.server import bp as server_bp
from blueprints.client import bp as client_bp

from flask_cors import CORS


app=Flask(__name__)
app.config.from_object(config)
app.json.ensure_ascii = False
db.init_app(app)
allowed_origins = ['http://localhost:5173','http://localhost:8080',]

CORS(app, supports_credentials=True,origins=allowed_origins,methods=["GET", "POST"], allow_headers=["Content-Type", "X-Token"])
with app.app_context():
    db.create_all()


app.register_blueprint(file_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(test_bp)
app.register_blueprint(resources_bp)
app.register_blueprint(server_bp)
app.register_blueprint(client_bp)

no_use_url=['/auth/login?','/auth/register?','/auth/check?','/auth/logout?','/captcha?']

@app.before_request
def myRequest():
    userID = session.get('userID')
    if userID:
        user = User.query.filter_by(id=userID).first()
        setattr(g, 'user', user)
    else:
        setattr(g, 'user', None)
    full_path=request.full_path
    if full_path not in no_use_url:
        if not userID:
           return jsonify({
              "code": 200,
              "message": "用户未登录！",
              "data": {}})

@app.context_processor
def my_context_processor():
    return {
        'user':g.user
    }



if __name__ =='__main__':
    app.run(debug = True)