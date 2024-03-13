from exts import db
from datetime import datetime
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False,unique=True)
    password = db.Column(db.String(100), nullable=False)

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

