import os
import time
import uuid
import tempfile
from flask import Flask, request, jsonify
import redis
import json
from flask_cors import CORS
import base64

def safe_base64_decode(data):
        # Strip prefix
        if "," in data:
            data = data.split(",")[1]

        # Fix padding
        missing_padding = len(data) % 4
        if missing_padding != 0:
            data += "=" * (4 - missing_padding)

        return base64.b64decode(data)

def create_app(redis_host='redis', redis_port=6379, redis_db=0):
    # Configure Flask application
    app = Flask(__name__)

    # Configure upload folder
    UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "./image_api_uploads")
    RESULTS_FOLDER = os.path.join(tempfile.gettempdir(), "./image_api_results")

    # Create directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_FOLDER, "iterations"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_FOLDER, "final_output"), exist_ok=True)
    
    # Configure Redis
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

    
    
    @app.route('/process', methods=['POST'])
    def process_image():
        try:
            # Parse JSON request body
            data = request.json
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
                
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
                
            # Get base64-encoded image from request
            base64_image = data.get('image')
            
            if not base64_image:
                return jsonify({'error': 'Empty image data'}), 400
                
            # Check if filename is provided
            filename = data.get('filename', 'image.jpg')
            if not filename:
                return jsonify({'error': 'Filename is required'}), 400
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            app.logger.info(f"Generated session ID: {session_id}")

            # Get parameters
            invert = data.get('invert', False)
            max_iterations = int(data.get('max_iterations', 10))
            initial_threshold = int(data.get('initial_threshold', 150))
            
            # Create base name from filename
            base_name = os.path.splitext(filename)[0]

            # No need to save to disk - pass directly to Redis
            input_path = f"{session_id}_{filename}"  # Just for reference
            
            # Create job data
            job_data = {
                'session_id': session_id,
                'filename': filename,
                'base_name': base_name,
                'input_path': input_path,
                'image_base64': base64_image,
                'invert': str(invert),
                'max_iterations': str(max_iterations),
                'initial_threshold': str(initial_threshold),
                'status': 'pending',
                'timestamp': str(time.time())
            }

            # Save job data to Redis
            redis_client.hset(f"job:{session_id}", mapping=job_data)
            redis_client.expire(f"job:{session_id}", 86400)
            
            # Enqueue the job
            app.logger.info(f"Enqueuing job for session ID: {session_id}")
            redis_client.lpush("image_processing_queue", json.dumps({
                'session_id': session_id,
                'job_data': job_data
            }))
            # Check push result
            app.logger.info(f"Redis lpush result")

            # Verify queue length
            queue_length = redis_client.llen("image_processing_queue")
            app.logger.info(f"Queue length after push: {queue_length}")
            
            # Return the job ID
            return jsonify({
                'status': 'success',
                'message': 'Image processing job submitted',
                'session_id': session_id
            })
            
        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/status/<session_id>', methods=['GET'])
    def get_job_status(session_id):
        """
        Get the status of a job
        """
        try:
            app.logger.info("start")
            job_key = f"job:{session_id}"
        
            # Log key details for debugging
            app.logger.info(f"Checking result key: {job_key}")
            
            # Check the type of the key first
            key_type = redis_client.type(job_key).decode('utf-8')
            app.logger.info(f"Key type: {key_type}")
            
            if key_type == 'string':
                job_data = json.loads(redis_client.get(job_key))
                return jsonify({
                    'status': job_data.get('status'),
                    'message': 'Job is still processing'
                })
            elif key_type == 'hash':
                # If it's a hash, get all fields
                status = redis_client.hget(job_key, 'status')
                if status is None:
                    return jsonify({
                        'status': 'not_found',
                        'message': 'Job status not found'
                    }), 404
                
                # Decode status
                status = status.decode('utf-8')
                job_data = redis_client.hgetall(job_key)
                # Convert byte keys/values to strings
                processed_job_data = {
                    k.decode('utf-8'): v.decode('utf-8') 
                    for k, v in job_data.items()
                }
                return jsonify({
                'status': status,
                'job_info': processed_job_data,
            })
            elif key_type != 'none':
                return jsonify({
                    'status': 'error',
                    'message': f'Job data has unexpected format: {key_type}'
                })
            
            return jsonify({
                'status': 'not_found',
                'message': 'Job not found'
            }), 404
        
        except Exception as e:
            app.logger.error(f"Error checking job status: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        # Check if Redis is reachable
        try:
            redis_client.ping()
            redis_status = 'connected'
        except Exception as e:
            redis_status = f'error: {str(e)}'
            
        return jsonify({
            'status': 'healthy',
            'redis': redis_status
        })
    
    return app

if __name__ == '__main__':
    # Get Redis config from environment variables
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_db = int(os.environ.get('REDIS_DB', 0))
    
    app = create_app(redis_host, redis_port, redis_db)
    # CORS(app, origins=[os.environ.get("CORS_ORIGIN", "http://localhost:3000")])
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))