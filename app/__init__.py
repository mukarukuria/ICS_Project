from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model

model = load_model('model/unet_model.h5')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db' 
app.config['UPLOAD_FOLDER'] = 'app\\static\\uploads'
app.config['SECRET_KEY'] = 'SHKEY'
db = SQLAlchemy(app)

from flask_login import LoginManager

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

from app import routes

import logging
logging.basicConfig(level=logging.INFO)