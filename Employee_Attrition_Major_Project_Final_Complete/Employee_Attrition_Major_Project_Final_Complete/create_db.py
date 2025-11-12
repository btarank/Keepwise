# create_db.py — run this from the project root where app.py is
from app import app, db
with app.app_context():
    db.create_all()
    print("DB created:", db.engine.url)
