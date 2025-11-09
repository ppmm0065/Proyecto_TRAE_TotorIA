# mi_aplicacion/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class FollowUp(db.Model):
    __tablename__ = 'follow_ups'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    related_filename = db.Column(db.String(255))
    related_prompt = db.Column(db.Text)
    related_analysis = db.Column(db.Text)
    follow_up_comment = db.Column(db.Text, nullable=False)
    follow_up_type = db.Column(db.String(50), default='general_comment')
    related_entity_type = db.Column(db.String(50))
    related_entity_name = db.Column(db.String(255))

class ConsumoTokensDiario(db.Model):
    __tablename__ = 'consumo_tokens_diario'
    # Hacemos una clave primaria compuesta simple para SQLite
    id = db.Column(db.Integer, primary_key=True)
    fecha = db.Column(db.Date, default=datetime.utcnow)
    modelo = db.Column(db.String(100))
    tokens_subida = db.Column(db.Integer, default=0)
    tokens_bajada = db.Column(db.Integer, default=0)
    costo_total = db.Column(db.Float, default=0.0)