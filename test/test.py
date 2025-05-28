import requests
import base64

# Endpoint URL
url = "http://127.0.0.1:8080/process"

# Files and form data
files = {"image": open("./test06.jpg", "rb")}
data = {
    "initial_threshold": "170",
    "invert": "true"  # Still passed, but now both versions are returned
}

# Send POST request
response = requests.post(url, files=files, data=data)

# Handle response
if response.status_code == 200:
    result = response.json()
    
    # Decode base64 images
    bw_image_data = base64.b64decode(result["bw_image_base64"])
    inverted_image_data = base64.b64decode(result["inverted_image_base64"])
    
    # Write images to disk
    with open("./output_bw.png", "wb") as f:
        f.write(bw_image_data)
    
    with open("./output_inverted.png", "wb") as f:
        f.write(inverted_image_data)
    
    print("Images processed and saved successfully!")
    print(f"Threshold used: {result['final_threshold']}, Iterations: {result['iterations']}, Acceptable: {result['acceptable']}")
else:
    print(f"Error: {response.status_code}, {response.text}")