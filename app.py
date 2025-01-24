from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load the model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    inputs = {
        'company': '',
        'type': '',
        'ram': '',
        'weight': '',
        'touchscreen': '',
        'ips': '',
        'screen_size': '',
        'resolution': '',
        'cpu': '',
        'hdd': '',
        'ssd': '',
        'gpu': '',
        'os': ''
    }

    if request.method == 'POST':
        # Get form inputs
        inputs['company'] = request.form['company']
        inputs['type'] = request.form['type']
        inputs['ram'] = int(request.form['ram'])
        inputs['weight'] = float(request.form['weight'])
        inputs['touchscreen'] = request.form['touchscreen']
        inputs['ips'] = request.form['ips']
        inputs['screen_size'] = float(request.form['screen_size'])
        inputs['resolution'] = request.form['resolution']
        inputs['cpu'] = request.form['cpu']
        inputs['hdd'] = int(request.form['hdd'])
        inputs['ssd'] = int(request.form['ssd'])
        inputs['gpu'] = request.form['gpu']
        inputs['os'] = request.form['os']

        # Calculate Pixels Per Inch (PPI)
        X_res, Y_res = map(int, inputs['resolution'].split('x'))
        ppi = ((X_res**2 + Y_res**2)**0.5) / inputs['screen_size']

        # Prepare query for prediction
        touchscreen = 1 if inputs['touchscreen'] == 'Yes' else 0
        ips = 1 if inputs['ips'] == 'Yes' else 0
        query = np.array([
            inputs['company'], inputs['type'], inputs['ram'], inputs['weight'], touchscreen, ips, ppi,
            inputs['cpu'], inputs['hdd'], inputs['ssd'], inputs['gpu'], inputs['os']
        ])
        query = query.reshape(1, -1)

        # Make prediction
        predicted_price = np.exp(pipe.predict(query)[0])  # Use np.exp if your model predicts log(price)

        # Format result
        result = f"The predicted price of this configuration is ₹{int(predicted_price):,}"

    # Render the HTML template with result and inputs
    return render_template('index.html', companies=df['Company'].unique(),
                           types=df['TypeName'].unique(), cpus=df['Cpu brand'].unique(),
                           gpus=df['Gpu brand'].unique(), oss=df['os'].unique(),
                           result=result, inputs=inputs)

if __name__ == '__main__':
    app.run(debug=True)
