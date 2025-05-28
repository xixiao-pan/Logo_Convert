import requests
import base64
import os
import time
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function
from utils import decode_base64_image, encode_image_to_base64


class ImageProcessingTest:
    def __init__(self, base_url="http://34.122.242.153/"):
        self.base_url = base_url
        self.test_image_path = "./test06.jpg"
        self.output_dir = "./test_outputs"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image(self, initial_threshold=150, invert=False):
        """
        Process an image and track its status
        """
        # Prepare request data
        base64_image = encode_image_to_base64(self.test_image_path)
        filename = os.path.basename(self.test_image_path)
        # Prepare request data
        json_data = {
            "image": base64_image,
            "filename": filename,
            "max_iterations": 5,
            "initial_threshold": initial_threshold,
            "invert": invert
        }


        # Submit job
        try:
            response = requests.post(
                f"{self.base_url}/process", 
                json=json_data,
                headers={"Content-Type": "application/json"}
            )
                
            if response.status_code != 200:
                print(f"Job submission failed: {response.status_code}")
                print(response.text)
                return None

            # Get session ID from response
            result = response.json()
            session_id = result.get('session_id')
            
            if not session_id:
                print("No session ID received")
                return None

            return session_id

        except Exception as e:
            print(f"Error submitting job: {e}")
            return None

    def check_job_status(self, session_id, max_wait=120):
        """
        Check job status with timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                # Check job status
                status_response = requests.get(f"{self.base_url}/status/{session_id}")
                
                if status_response.status_code != 200:
                    print(f"Status check failed: {status_response.status_code}")
                    return None

                status_data = status_response.json()
                status = status_data.get('job_info', {}).get('status')
                
                print(f"Current status: {status}")

                # Check if job is completed
                if status == 'completed':
                    return status_data

                # Wait before next check
                time.sleep(2)

            except Exception as e:
                print(f"Error checking status: {e}")
                return None

        print("Job status check timed out")
        return None

    def get_job_result(self, session_id):
        """
        Retrieve job result
        """
        try:
            result_response = requests.get(f"{self.base_url}/status/{session_id}")
            
            if result_response.status_code != 200:
                print(f"Result retrieval failed: {result_response.status_code}")
                return None
            decode_base64_image(result_response.json().get('job_info').get('bw_image_base64'), os.path.join(self.output_dir, f"result_{session_id}.png"))
            return result_response.json()

        except Exception as e:
            print(f"Error retrieving result: {e}")
            return None

    def run_full_test(self, initial_threshold=150, invert=False):
        """
        Run complete image processing test
        """
        print(f"Starting image processing test (threshold={initial_threshold}, invert={invert})")
        
        # Submit job
        session_id = self.process_image(
            initial_threshold=initial_threshold, 
            invert=invert
        )
        
        if not session_id:
            print("Job submission failed")
            return False

        print(f"Job submitted with session ID: {session_id}")

        # Wait for job completion
        job_status = self.check_job_status(session_id)
        
        if not job_status:
            print("Job did not complete successfully")
            return False

        # Retrieve full job result
        job_result = self.get_job_result(session_id)
        
        if not job_result:
            print("Could not retrieve job result")
            return False

        # Print job details
        print("Job Details:")
        print(job_result)

        return True

def main():
    # Create test instance
    test = ImageProcessingTest()

    # Run tests with different parameters
    tests = [
        {"initial_threshold": 150, "invert": False},
        {"initial_threshold": 170, "invert": True}
    ]

    # Run multiple test scenarios
    for test_config in tests:
        success = test.run_full_test(**test_config)
        print(f"Test {'PASSED' if success else 'FAILED'}\n")

if __name__ == "__main__":
    main()