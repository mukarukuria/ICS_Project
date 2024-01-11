from app import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    gender = db.Column(db.String(10))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(80))
    cases = db.relationship('Case', backref='user', lazy=True)
    results = db.relationship('Result', backref='case', lazy=True)

class Case(db.Model):
    caseid = db.Column(db.Integer, primary_key=True)
    case_name = db.Column(db.String(120))
    notes = db.Column(db.String(500))
    file_path = db.Column(db.String(120))
    note = db.relationship('Note', backref='result', lazy=True)
    submitted_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Result(db.Model):
    result_id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.String(120))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    case_id = db.Column(db.Integer, db.ForeignKey('case.caseid'), nullable=False)

class Note(db.Model):
    notes_id = db.Column(db.Integer, primary_key=True)
    additional_notes = db.Column(db.String(500))
    case_id = db.Column(db.Integer, db.ForeignKey('case.caseid'), nullable=False)