from prometheus_client import start_http_server, Summary

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

@app.route('/predict', methods=['POST'])
@REQUEST_TIME.time()
def predict():
    ...
