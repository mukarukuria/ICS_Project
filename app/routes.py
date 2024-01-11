from flask import render_template, request, redirect, url_for, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from app import model
import pydicom
import numpy as np
from app import app, db
from app.models import User, Case, Result, Note
from PIL import Image
import os

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    cases = Case.query.all()
    return render_template('index.html', name=session['name'], cases=cases)

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        gender = request.form.get('gender')
        email = request.form.get('email')
        password = request.form.get('password')
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "A user with that email already exists"
        user = User(name=name, gender=gender, email=email, password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('home'))
    return render_template('authentication.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                session['name'] = user.name
                login_user(user)
                return redirect(url_for('home'))
            else:
                return "Invalid password"
        else:
            return "Invalid email"
    return render_template('authentication.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/results/<int:caseid>')
@login_required
def results(caseid):
    case = Case.query.get(caseid)
    notes = Note.query.filter_by(case_id=caseid).all()
    result = Result.query.filter_by(case_id=caseid).first() 
    png_filename = case.file_path.replace('.dcm', '.png')    
    return render_template('results.html', case=case, png_filename=png_filename, notes=notes, result=result)

@app.route('/submit_case', methods=['POST'])
@login_required
def submit_case():
    case_name = request.form['case_name']
    notes = request.form['notes']
    file = request.files['file']

    if file and file.filename.endswith('.dcm'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        case = Case(case_name=case_name, notes=notes, file_path=filename, submitted_by=current_user.id)
        db.session.add(case)
        db.session.commit()

        ds = pydicom.dcmread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = ds.pixel_array

        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        image_file = Image.fromarray((image*255).astype(np.uint8))
        png_filename = filename.replace('.dcm', '.png')
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], png_filename))
        image = image.reshape(1, image.shape[0], image.shape[1], 1)
        image = np.moveaxis(image, 0, -1)

        mask = model.predict(image[np.newaxis, ...])[0]
        mask = (mask > 0.5).astype(np.uint8)
        mask = np.squeeze(mask)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        
        mask_filename = filename.replace('.dcm', '_mask.png')
        mask_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
        mask_image.save(mask_filepath)

        result = Result(case_id=case.caseid, user_id=current_user.id, file_path=mask_filename)
        db.session.add(result)
        db.session.commit()

        return redirect(url_for('results', caseid=case.caseid))

    return "Invalid file"

@app.route('/add_notes', methods=['POST'])
@login_required
def add_notes():
    additional_notes = request.form.get('additional_notes')
    case_id = request.form.get('case_id')
    print(f"Additional Notes: {additional_notes}, Case ID: {case_id}")
    note = Note(additional_notes=additional_notes, case_id=case_id)
    db.session.add(note)
    db.session.commit()
    return redirect(url_for('results', caseid=case_id))

@app.route('/delete_case/<int:caseid>', methods=['POST'])
@login_required
def delete_case(caseid):
    case = Case.query.get(caseid)
    Note.query.filter_by(case_id=caseid).delete()
    db.session.delete(case)
    db.session.commit()
    return redirect(url_for('home'))