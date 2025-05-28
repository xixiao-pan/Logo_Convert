import os
import time
import json
import logging
import cv2
import numpy as np
import redis
import tempfile
from PIL import Image
from google import genai
from google.genai import types
import re
import signal
import sys
import shutil
import base64
from utils import encode_image_to_base64, decode_base64_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
)
logger = logging.getLogger('worker')


class ImageProcessingWorker:
    def __init__(self, redis_host='redis', redis_port=6379, redis_db=0):
        # Configure Redis
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        
        # Configure directories
        self.UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "./image_api_uploads")
        self.RESULTS_FOLDER = os.path.join(tempfile.gettempdir(), "./image_api_results")
        
        # Create directories
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.RESULTS_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(self.RESULTS_FOLDER, "iterations"), exist_ok=True)
        os.makedirs(os.path.join(self.RESULTS_FOLDER, "final_output"), exist_ok=True)
        
        # Configure Gemini API
        api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAUj5lmD9jZRSKMIvfSRHAMKsS7fDQrABw")
        self.client = genai.Client(api_key=api_key)
        
        logger.info("Worker initialized successfully")


    def get_gemini_feedback(self, original_image, processed_image_path, max_retries=3):
        """
        Use Gemini AI to analyze original and processed images, get improvement suggestions
        Will automatically retry if unable to get JSON formatted response
        """
        # Read original and processed images
        original_image = Image.open(original_image)
        processed_image = Image.open(processed_image_path)
        
        # Create prompt
        prompt = (
            "I'm working on creating a black and white logo from the original image. The background should be white. "
            "The first image is the original, and the second is my processed version with black and white colors. "
            "Please analyze the processed version and tell me what specific patterns or elements have been lost in the conversion and whether these areas should be black or white. "
            "Make sure the original image frame is not changed."
            "The processed image is acceptable as long as important features like the figures or patterns are included. "
            "Set acceptable to true if the processed image captures most shades and details and the essential elements of the logo. "
            "Use false if it does not or contain scattered dots. "
            "IMPORTANT: Format your response ONLY as a valid JSON dictionary with two fields: "
            "'feedback' containing your detailed analysis and "
            "'acceptable' with a boolean value (true/false) indicating whether the processed image is acceptable as is. "
            "Example format: {\"feedback\": \"Your detailed analysis here...\", \"acceptable\": true}"
        )
        
        # Retry mechanism
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to get feedback (attempt {attempt+1}/{max_retries})...")
                
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[prompt, original_image, processed_image],
                    config=types.GenerateContentConfig(
                        response_modalities=['Text'],
                    )
                )
                
                # Extract text feedback
                response_text = response.candidates[0].content.parts[0].text
                logger.info(f"Got original response: {response_text[:100]}...")  # Print only first 100 chars
                
                # Parse JSON response
                # Method 1: Check if contains code block markers
                if "```json" in response_text:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                        try:
                            result = json.loads(json_content)
                            logger.info("Successfully parsed JSON from code block")
                            return result
                        except json.JSONDecodeError:
                            logger.info("Code block parsing failed, trying other methods")
                
                # Method 2: Try parsing entire response directly
                try:
                    result = json.loads(response_text)
                    logger.info("Successfully parsed JSON directly")
                    return result
                except json.JSONDecodeError:
                    logger.info("Direct parsing failed, trying to find JSON object")
                
                # Method 3: Find part surrounded by braces
                json_pattern = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_pattern:
                    try:
                        json_content = json_pattern.group(0)
                        result = json.loads(json_content)
                        logger.info("Successfully extracted JSON object from text")
                        return result
                    except json.JSONDecodeError:
                        logger.info("JSON object extraction failed")
                
                # All parsing methods failed, retry if not last attempt
                if attempt < max_retries - 1:
                    logger.info("Unable to parse JSON, preparing to retry...")
                    # Modify prompt to more explicitly require JSON format
                    prompt = (
                        "I'm working on creating a black and white logo from the original image. "
                        "The first image is the original, and the second is my processed version with black and white colors. "
                        "CRITICAL: You must respond ONLY with a valid JSON object in the exact format: "
                        "{\"feedback\": \"your analysis\", \"acceptable\": true or false} "
                        "Do not include any other text, markdown formatting, or code blocks. "
                        "Please analyze the processed version and tell me in the feedback field: "
                        "1. What specific patterns or elements have been lost in the conversion? "
                        "2. Should these areas be black or white? "
                        "Set acceptable to true if the processed image captures most shades and details and the essential elements of the logo. Use false if it does not or contain scattered dots. "
                    )
                    time.sleep(1)  # Brief pause before retrying
                else:
                    # Last attempt failed, return default response
                    logger.warning("Warning: Unable to get JSON formatted response after multiple attempts, using original response")
                    return {
                        "feedback": response_text,
                        "acceptable": False
                    }
                    
            except Exception as e:
                logger.error(f"Error getting feedback on attempt {attempt+1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    return {
                        "feedback": f"Error: Failed after multiple attempts - {str(e)}",
                        "acceptable": False
                    }

    def get_threshold_recommendation(self, original_image, processed_image_path, feedback, crt_threshold=128):
        """
        Get new threshold recommendation based on current processing result
        """
        # Read original and processed images
        original_image = Image.open(original_image)
        processed_image = Image.open(processed_image_path)
        
        # Ensure feedback is a dictionary type, and safely extract feedback content
        feedback_text = ""
        if isinstance(feedback, dict):
            feedback_text = str(feedback.get("feedback", ""))
        else:
            feedback_text = str(feedback)
        
        prompt = (
            f"I'm working on creating a black and white logo from the original image. "
            f"The first image is the original, and the second is my processed version with black and white colors. "
            f"I have analyzed the processed version and gotten this feedback: \"{feedback_text}\". "
            f"Based on this feedback, there are details omitted after processing. Please compare the original with the processed image, "
            f"what threshold value (between 0-255) should I use for cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY) "
            f"to make sure almost no detail is lost with a better black and white logo. Currently I am using {crt_threshold}. "
            f"The threshold means the gray value above which will be converted to black, and below to white. "
            f"Please respond with just a number representing the optimal threshold value."
        )
        
        # Call API to get recommendation
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[prompt, original_image, processed_image],
                config=types.GenerateContentConfig(
                    response_modalities=['Text'],
                )
            )
            
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    response_text = part.text.strip()
                    logger.info(f"Threshold recommendation response: {response_text}")
                    
                    # Try to extract number
                    threshold_match = re.search(r'\b(\d+)\b', response_text)
                    if threshold_match:
                        threshold = int(threshold_match.group(1))
                        # Ensure threshold is in valid range
                        threshold = max(0, min(255, threshold))
                        logger.info(f"Extracted threshold recommendation: {threshold}")
                        return threshold
            
            raise ValueError("Unable to extract threshold from response")
        
        except Exception as e:
            logger.error(f"Error getting threshold recommendation: {str(e)}")
            raise e

    def invert_colors(self, image_path, output_path):
        """
        Invert image colors - change black to white and white to black, preserving transparency
        """
        # Load image with transparency channel
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # Robust alpha channel detection
        has_alpha = False
        if len(image.shape) == 4:
            has_alpha = True  # 4-channel image (BGRA)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            has_alpha = True  # 3D array but with 4 channels (BGRA)
        
        logger.info(f"Image shape: {image.shape}")
        logger.info(f"Contains alpha channel: {has_alpha}")
        
        try:
            if has_alpha:
                logger.info(f"Detected image with transparency, will preserve alpha channel when inverting colors")
                
                # Separate color channels and alpha channel
                if len(image.shape) == 4:
                    alpha_channel = image[:, :, :, 3:4]
                    color_channels = image[:, :, :, 0:3]
                else:
                    alpha_channel = image[:, :, 3:4]
                    color_channels = image[:, :, 0:3]
                
                # Only invert color channels
                inverted_colors = 255 - color_channels
                
                # Merge inverted color channels and original alpha channel
                if len(image.shape) == 4:
                    inverted_image = np.concatenate((inverted_colors, alpha_channel), axis=3)
                else:
                    # Create empty image first
                    inverted_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                    # Fill with inverted colors
                    inverted_image[:, :, 0:3] = inverted_colors
                    # Fill with original transparency
                    inverted_image[:, :, 3] = image[:, :, 3]
            else:
                # Regular inversion (no alpha channel)
                inverted_image = 255 - image
        
        except Exception as e:
            logger.error(f"Color inversion error: {str(e)}")
            logger.info(f"Using simple method to invert colors...")
            
            # Use simple method if failed
            try:
                if len(image.shape) == 3 and image.shape[2] == 4:
                    # Handle color and alpha channels separately
                    alpha = image[:, :, 3]
                    rgb = image[:, :, 0:3]
                    
                    # Invert RGB channels
                    inverted_rgb = 255 - rgb
                    
                    # Merge back
                    inverted_image = np.zeros_like(image)
                    inverted_image[:, :, 0:3] = inverted_rgb
                    inverted_image[:, :, 3] = alpha
                else:
                    # No transparency channel or other cases
                    inverted_image = 255 - image
            except:
                # If still fails, use pixel-by-pixel operation
                inverted_image = np.copy(image)
                if len(image.shape) == 3 and image.shape[2] >= 3:
                    for i in range(3):  # Only invert first three channels (BGR)
                        inverted_image[:, :, i] = 255 - image[:, :, i]
                else:
                    inverted_image = 255 - image
        
        # Save processed image
        cv2.imwrite(output_path, inverted_image)
        logger.info(f"Color-inverted image saved to: {output_path}")
        
        return output_path

    

    def iterative_image_processing(self, image, session_id, base_name, max_iterations=3, initial_threshold=150):
        """
        Iteratively process image: process -> get feedback -> adjust threshold -> process again
        
        Args:
            image_path: Original image path
            session_id: Unique session ID for this processing job
            base_name: Image base name (without extension)
            max_iterations: Maximum number of iterations
            initial_threshold: Initial threshold value
        """
        # Create iteration subdirectory
        iterations_dir = os.path.join(self.RESULTS_FOLDER, "iterations", session_id)
        os.makedirs(iterations_dir, exist_ok=True)
        
        # Set output paths
        final_output_path = os.path.join(self.RESULTS_FOLDER, "final_output", f"{session_id}_{base_name}_final.png")
        inverted_output_path = os.path.join(self.RESULTS_FOLDER, "final_output", f"{session_id}_{base_name}_inverted.png")
        
        # Initialize variables
        current_threshold = initial_threshold
        current_processed_path = None
        acceptable = False
        
        # Iterative processing
        for iteration in range(max_iterations):
            iteration_num = iteration + 1
            logger.info(f"Iteration {iteration_num}/{max_iterations}: Threshold = {current_threshold}")
            
            # Process image
            current_processed_path = os.path.join(iterations_dir, f"iteration_{iteration_num}.png")
            self.convert_to_black_white(current_threshold, image, current_processed_path)
            
            # Get feedback
            feedback = self.get_gemini_feedback(image, current_processed_path)
            
            # Check if result is acceptable
            acceptable = feedback.get("acceptable", False)
            if acceptable:
                logger.info(f"Image processing result is acceptable, stopping iterations")
                break
            
            # If there's another iteration, get new threshold
            if iteration < max_iterations - 1:
                # Get new threshold recommendation
                logger.info(f"Getting new threshold recommendation...")
                try:
                    new_threshold = self.get_threshold_recommendation(
                        image, current_processed_path, feedback, current_threshold
                    )
                    
                    # Update threshold
                    current_threshold = new_threshold
                    
                except Exception as e:
                    logger.error(f"Error getting new threshold: {str(e)}")
                    logger.info(f"Continuing with current threshold: {current_threshold}")
        
        # Process finished, save results
        if current_processed_path:
            import shutil
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            shutil.copy(current_processed_path, final_output_path)
            
            # Create inverted version
            self.invert_colors(final_output_path, inverted_output_path)
            
            logger.info(f"Processing complete: Final threshold = {current_threshold}, Acceptable = {acceptable}")
            
        # Encode images to base64
        bw_image_base64 = encode_image_to_base64(final_output_path)
        inverted_image_base64 = encode_image_to_base64(inverted_output_path)
        logger.info(f"Encoded images to base64")
        return {
            # "final_threshold": current_threshold,
            # "acceptable": acceptable,
            # "iterations": iteration + 1,
            "final_path": final_output_path,
            "inverted_path": inverted_output_path,
            "bw_image_base64": bw_image_base64,
            "inverted_image_base64": inverted_image_base64,
            # "status": "completed"
        }
    
    def process_job(self, job_data):
        """
        Process a job from the queue
        """
        logger.info(f"Received job data: {job_data}")
        session_id = job_data.get('session_id')
        if not session_id:
            logger.error("Job data missing session_id")
            return
            
        logger.info(f"Processing job {session_id}")
        # logger.info(session_id.type())
        try:
            # Update job status
            self.redis_client.hset(f"job:{session_id}", mapping={
                **job_data,  # Preserve original data
                'status': 'processing',
                'started_at': str(time.time())
            })
            logger.info(f"test2")
            # logger.info(f"test2 Job {session_id} status updated to processing")
            # Get job parameters
            input_path = job_data.get('input_path')
            base_name = job_data.get('base_name')
            max_iterations = job_data.get('max_iterations', 3)
            initial_threshold = job_data.get('initial_threshold', 150)
            logger.info(f"test1")
            data = job_data.get('image_base64')
            # logger.info(f"length: {len(data)}")
            # # Strip prefix
            if "," in data:
                data = data.split(",")[1]

            # Fix padding
            missing_padding = len(data) % 4
            logger.info(f"Missing padding: {missing_padding}, {len(data)}")
            if missing_padding != 0:
                logger.info(f"Missing padding detected: {missing_padding} bytes")
                data += "=" * (4 - missing_padding)

            image_base64 = base64.b64decode(data)
            image = decode_base64_image(data)
            
            # Process image
            result = self.iterative_image_processing(
                image=image,
                session_id=session_id,
                base_name=base_name,
                max_iterations=int(max_iterations),
                initial_threshold=int(initial_threshold)
            )
            logger.info(f"Job {session_id} processing result")
            # Save result to Redis
            self.redis_client.hset(f"job:{session_id}", mapping={
                **job_data,  # Preserve original data
                **result,
                'status': 'completed',
                'completed_at': str(time.time())
            })
            
            logger.info(f"Job {session_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing job {session_id}: {str(e)}")
            # Update job status
            error_result = {
                "status": "error",
                "error": str(e)
            }
            self.redis_client.hset(f"job:{session_id}", mapping={
                **job_data,
                'status': 'error',
                "error": str(e)})
            return error_result

    def convert_to_black_white(self, thre_value, image, output_path="processed_image.png"):
        """Process image locally to black and white effect, supporting transparent PNG images"""
        # Load image with transparency channel
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        
        # Robust alpha channel detection
        has_alpha = False
        if len(image.shape) == 4:
            has_alpha = True  # 4-channel image (BGRA)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            has_alpha = True  # 3D array but with 4 channels (BGRA)
        
        logger.info(f"Image shape: {image.shape}")
        logger.info(f"Contains alpha channel: {has_alpha}")
        
        if has_alpha:
            logger.info(f"Detected image with transparency, will preserve alpha channel")
            try:
                # Separate color channels and alpha channel
                if len(image.shape) == 4:
                    bgr_channels = image[:, :, :, 0:3]
                    alpha_channel = image[:, :, :, 3]
                else:
                    bgr_channels = image[:, :, 0:3]
                    alpha_channel = image[:, :, 3]
                
                # Convert to grayscale
                gray = cv2.cvtColor(bgr_channels, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur
                gray = cv2.GaussianBlur(gray, (9, 9), 0)
                
                # Apply threshold
                _, mask = cv2.threshold(gray, thre_value, 255, cv2.THRESH_BINARY)
                
                # Clean mask through erosion and dilation
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                # Create an empty three-channel image for storing black and white results
                processed_bgr = np.zeros_like(bgr_channels)
                
                # Apply mask to each channel
                # White areas (255,255,255)
                white_mask = (mask == 255)
                if len(bgr_channels.shape) == 3:
                    processed_bgr[white_mask, 0] = 255
                    processed_bgr[white_mask, 1] = 255
                    processed_bgr[white_mask, 2] = 255
                else:
                    for c in range(3):
                        processed_bgr[white_mask, c] = 255
                
                # Merge processed bgr image and original alpha channel
                if len(image.shape) == 4:
                    final_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], 4), dtype=np.uint8)
                    final_image[:, :, :, 0:3] = processed_bgr
                    final_image[:, :, :, 3] = alpha_channel
                else:
                    final_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                    final_image[:, :, 0:3] = processed_bgr
                    final_image[:, :, 3] = alpha_channel
            
            except Exception as e:
                # If processing fails, try alternative method
                logger.error(f"Alpha processing error: {str(e)}")
                logger.info(f"Using alternative method for image with transparency...")
                
                # Convert to grayscale (only process RGB channels)
                bgr = image[:, :, :3] if image.shape[2] == 4 else image
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur
                gray = cv2.GaussianBlur(gray, (9, 9), 0)
                
                # Apply threshold
                _, bw = cv2.threshold(gray, thre_value, 255, cv2.THRESH_BINARY)
                
                # Create black and white RGB image
                bw_rgb = np.zeros(bgr.shape, dtype=np.uint8)
                bw_rgb[:, :, 0] = bw
                bw_rgb[:, :, 1] = bw
                bw_rgb[:, :, 2] = bw
                
                # Keep original alpha channel
                alpha = image[:, :, 3] if image.shape[2] == 4 else np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
                
                # Merge
                final_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                final_image[:, :, :3] = bw_rgb
                final_image[:, :, 3] = alpha
        else:
            # Regular processing (no alpha channel)
            logger.info(f"Standard image processing (no transparency channel)")
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur
                gray = cv2.GaussianBlur(gray, (9, 9), 0)
                
                # Apply threshold
                _, mask = cv2.threshold(gray, thre_value, 255, cv2.THRESH_BINARY)
                
                # Clean mask through erosion and dilation
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                # Expand mask to 3 channels
                mask_3ch = cv2.merge([mask, mask, mask])
                
                # Use mask to directly create black and white image (safer method)
                final_image = np.zeros_like(image)
                final_image[mask_3ch > 0] = 255
            
            except Exception as e:
                # If processing fails, try alternative method
                logger.error(f"Standard processing error: {str(e)}")
                logger.info(f"Using alternative method to process image...")
                
                # Simple binarization processing
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # RGB image
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    # RGBA image
                    gray = cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_BGR2GRAY)
                else:
                    # Possibly already a grayscale image
                    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold
                _, bw = cv2.threshold(gray, thre_value, 255, cv2.THRESH_BINARY)
                
                # Create final image based on original image channel count
                if len(image.shape) == 3 and image.shape[2] == 4:
                    final_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                    final_image[:, :, 0] = bw
                    final_image[:, :, 1] = bw
                    final_image[:, :, 2] = bw
                    final_image[:, :, 3] = image[:, :, 3]  # Keep original alpha
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    final_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                    final_image[:, :, 0] = bw
                    final_image[:, :, 1] = bw
                    final_image[:, :, 2] = bw
                else:
                    final_image = bw
        
        # Save processed image
        cv2.imwrite(output_path, final_image)
        logger.info(f"Locally processed image saved to: {output_path}")
        
        return output_path


class GracefulKiller:
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, *args):
        self.kill_now = True
        logger.info("Received shutdown signal, will exit after current job completes")
        
def run_worker():
    """
    Main worker function to poll and process jobs from Redis queue
    """
    # Get Redis config from environment variables
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_db = int(os.environ.get('REDIS_DB', 0))
    
    # Initialize worker and signal handler
    worker = ImageProcessingWorker(redis_host, redis_port, redis_db)
    killer = GracefulKiller()
    
    logger.info(f"Worker started, connecting to Redis at {redis_host}:{redis_port}")
    
    while not killer.kill_now:
        try:
            # Try to get a job from the queue with a timeout
            result = worker.redis_client.brpop("image_processing_queue", timeout=1)
            # logging.info(f"Got job from queue: {result}")
            if result:
                logging.info(f"Got job from queue: {result}")
                _, job_data_bytes = result
                job_data = json.loads(job_data_bytes)
                worker.process_job(job_data.get('job_data', {}))
            
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection error: {str(e)}")
            time.sleep(5)  # Wait before retrying
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            time.sleep(1)  # Brief pause before continuing
    
    logger.info("Worker shutting down")
    
if __name__ == "__main__":
    run_worker()

