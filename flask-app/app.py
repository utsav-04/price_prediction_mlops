from flask import Flask,render_template,request,jsonify
import mlflow
import dagshub
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

mlflow.set_tracking_uri('https://dagshub.com/utsav-04/price_prediction_mlops.mlflow')
dagshub.init(repo_owner='utsav-04', repo_name='price_prediction_mlops', mlflow=True)

app = Flask(__name__)

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

brands = [
    "Google", "Honor", "Huawei", "IKall", "Infinix", "iQOO", "Itel", "Jio", "Lava", 
    "Letv", "LG", "Motorola", "Nokia", "OnePlus", "Oppo", "Poco", "Realme", "Redmi", 
    "Royole", "Samsung", "Sony", "Tecno", "Vivo", "Xiaomi", "ZTE"
]

def encode_brand(brand_name, brands_list):
    one_hot = [1 if brand == brand_name else 0 for brand in brands_list]
    return one_hot

scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():
    rating = float(request.form.get("rating", 0))
    processor_speed = float(request.form.get("processor_speed", 0))
    ram_capacity = float(request.form.get("ram_capacity", 0))
    battery_capacity = float(request.form.get("battery_capacity", 0))
    internal_memory = float(request.form.get("internal_memory", 0))
    screen_size = float(request.form.get("screen_size", 0))
    refresh_rate = float(request.form.get("refresh_rate", 0))
    num_rear_cameras = int(request.form.get("num_rear_cameras", 0))
    num_front_cameras = int(request.form.get("num_front_cameras", 0))
    brand = request.form.get("brand", "Unknown")
    has_ir_blaster = int(request.form.get("has_ir_blaster", 0) == "true")  # Convert 'true' to 1, 'false' to 0
    extended_memory_available = int(request.form.get("extended_memory_available", 0))
    extended_upto = float(request.form.get("extended_upto", 0))

        # One-hot encode the brand
    brand_one_hot = encode_brand(brand, brands)

    new_smartphone_features = [
            rating, processor_speed,battery_capacity , ram_capacity,
            internal_memory, screen_size, refresh_rate,
        num_rear_cameras, num_front_cameras,extended_memory_available, extended_upto
    ] + brand_one_hot + [
        has_ir_blaster, 
    ]

        # Convert to DataFrame for scaling
    new_smartphone_df = pd.DataFrame([new_smartphone_features])

        # Standardize the features
        #scaler = StandardScaler()
        # Standardize the features

    new_smartphone_scaled = scaler.transform(new_smartphone_df)

        # Predict price
    predicted_price = model.predict(new_smartphone_scaled)
        
    return render_template('index.html',result=f"{predicted_price[0]:.2f}")

if __name__ == '__main__':
    app.run(debug=True)