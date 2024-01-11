document.getElementById('login-tab').addEventListener('click', function() {
  document.getElementById('login-tab').classList.add('active');
  document.getElementById('register-tab').classList.remove('active');
  document.getElementById('login_form').style.display = 'block';
  document.getElementById('register_form').style.display = 'none';
});
  
document.getElementById('register-tab').addEventListener('click', function() {
  document.getElementById('register-tab').classList.add('active');
  document.getElementById('login-tab').classList.remove('active');
  document.getElementById('register_form').style.display = 'block';
  document.getElementById('login_form').style.display = 'none';
});

function validateLoginForm() {
    var email = document.getElementById('loginemail').value;
    var password = document.getElementById('loginpassword').value;

    if (email == "" || password == "") {
        alert("Email and password must be filled out");
        return false;
    }
    return true;
}

function validateRegisterForm() {
    var name = document.getElementById('name').value;
    var gender = document.getElementById('gender').value;
    var email = document.getElementById('registeremail').value;
    var password = document.getElementById('registerpassword').value;
    var confirm = document.getElementById('confirm').value;

    if (name == "") {
      alert("Name field must be filled out");
      return false;
  }
  if (email == "") {
    alert("Email must be filled out");
    return false;
  }
  if (password == "") {
    alert("Password must be filled out");
    return false;
  }
  if (confirm == "") {
    alert("Confirm Password must be filled out");
    return false;
  }
    if (password != confirm) {
        alert("Passwords do not match");
        return false;
    }

    return true;
}