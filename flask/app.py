from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle et le scaler
model = joblib.load('model/best_xgboost.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Recevoir les données (séquence de 10 timesteps)
    data = request.json['sequence']  # Format: [ [f1, f2, ...], ... ] (10x nb_features)
    
    # Prétraitement
    data_scaled = scaler.transform(np.array(data).reshape(-1, len(data[0])))
    
    # Prédiction
    prediction = model.predict(data_scaled)
    
    # Retourner le résultat
    return jsonify({
        "prediction": int(prediction[0]),
        "status": "success"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)