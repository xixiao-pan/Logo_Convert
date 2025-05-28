import base64

import os
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded

def decode_base64_image(base64_string, output_path=None):
    """
    Decode a base64 string to an image file.
    
    :param base64_string: Base64 encoded string of the image.
    :param output_path: Path to save the decoded image. If None, a default path will be generated.
    :return: Path of the saved image.
    """
    # if base64_string.startswith('data:image'):
    #     base64_string = base64_string.split(',')[1]
    
    # Decode base64 string
    image_data = base64.b64decode(base64_string)
    
    # Generate output path if not provided
    if output_path is None:
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        print("outputs directory created")
        output_path = os.path.join('outputs', f'decoded_image_.png')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the decoded image
    with open(output_path, 'wb') as file:
        file.write(image_data)
    
    print(f"Image saved successfully: {output_path}")
    return output_path