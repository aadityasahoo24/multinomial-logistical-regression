import kagglehub

# Download latest version
path = kagglehub.dataset_download("ananthr1/weather-prediction")

print("Path to dataset files:", path)