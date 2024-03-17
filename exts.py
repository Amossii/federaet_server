from flask_sqlalchemy import SQLAlchemy

from flask import Blueprint,request,session,g
from blueprints.modules.massage import packMassage

db=SQLAlchemy()
from torchcam.methods import GradCAM