from exts import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), nullable=False,unique=True)
    password = db.Column(db.String(255), nullable=False)
    join_time = db.Column(db.DateTime, default=datetime.now)
    email=db.Column(db.String(255))
    access=db.Column(db.Boolean, nullable=False)

    @property
    def _password(self):
        return self.password

    @_password.setter
    def _password(self, value):
        self.password = generate_password_hash(value)

    def _check_password(self, user_pad):
        return check_password_hash(self.password, user_pad)

class File(db.Model):
    __tablename__ = "file"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    content = db.Column(db.LargeBinary(16777216), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    join_time=db.Column(db.DateTime,default=datetime.now)
    file_size = db.Column(db.Integer, default=datetime.now)

    user_id=db.Column(db.Integer,db.ForeignKey("user.id"))
    user=db.relationship("User",backref="files")

class Client_model(db.Model):
    __tablename__ = "client"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    number=db.Column(db.Integer,unique=True)
    filename = db.Column(db.String(100), nullable=False)
    join_time = db.Column(db.DateTime, default=datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    user = db.relationship("User", backref="clients")

class Dpmodel_model(db.Model):
    __tablename__ = "model"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    content = db.Column(db.LargeBinary(16777216), nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    join_time = db.Column(db.DateTime, default=datetime.now)
    file_size = db.Column(db.Integer, default=0)
    acc=db.Column(db.Double,default=0)
    loss=db.Column(db.Double,default=0)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    user = db.relationship("User", backref="models")

