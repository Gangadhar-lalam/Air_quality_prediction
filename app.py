from flask import Flask, request, render_template, jsonify
import pandas as pd
import requests
import os

from dotenv import load_dotenv
load_dotenv()

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# Initialize Flask
app = Flask(__name__)


# ================= HOME PAGE =================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')


# ================= LIVE DATA API =================

@app.route("/get_live_data", methods=["POST"])
def get_live_data():

    try:

        data = request.get_json()

        lat = data["lat"]
        lon = data["lon"]

        API_KEY = os.getenv("API_KEY")

        if not API_KEY:
            return jsonify({"error": "API key not found"}), 500


        air_url = (
            f"https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={lat}&lon={lon}&appid={API_KEY}"
        )

        weather_url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        )


        air_res = requests.get(air_url).json()
        weather_res = requests.get(weather_url).json()


        # ✅ Check if API returned error
        if "list" not in air_res:
            print("Air API Error:", air_res)
            return jsonify({"error": "Air API failed"}), 500


        if "main" not in weather_res:
            print("Weather API Error:", weather_res)
            return jsonify({"error": "Weather API failed"}), 500


        components = air_res["list"][0]["components"]


        result = {

                "Temperature": weather_res["main"]["temp"],
                "Humidity": weather_res["main"]["humidity"],

                "PM2_5": components["pm2_5"],
                "PM10": components["pm10"],
                "NO2": components["no2"],
                "SO2": components["so2"],

                # ✅ FIX HERE
                "CO": components["co"] / 1000,   # convert μg → mg

                "Proximity_to_Industrial_Areas": 5,
                "Population_Density": 8000
}

        return jsonify(result)


    except Exception as e:

        print("Server Error:", e)

        return jsonify({"error": "Server error"}), 500
# ================= PREDICTION =================

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    else:

        # Get form data
        data = CustomData(

            Temperature=float(request.form.get('Temperature')),
            Humidity=float(request.form.get('Humidity')),

            PM2_5=float(request.form.get('PM2_5')),   # ✅ FIXED
            PM10=float(request.form.get('PM10')),

            NO2=float(request.form.get('NO2')),
            SO2=float(request.form.get('SO2')),
            CO=float(request.form.get('CO')),

            Proximity_to_Industrial_Areas=float(
                request.form.get('Proximity_to_Industrial_Areas')
            ),

            Population_Density=float(
                request.form.get('Population_Density')
            )
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()

        # Predict
        pipeline = PredictPipeline()
        result = pipeline.predict(pred_df)

        return render_template('home.html', results=result[0])


# ================= RUN SERVER =================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )