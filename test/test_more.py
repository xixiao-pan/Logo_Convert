import os
import sys
import time
import concurrent.futures
import requests
import json


# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import encode_image_to_base64, decode_base64_image

class AsyncImageProcessingTest:
    def __init__(self, base_url="http://34.122.242.153/"):
        self.base_url = base_url
        self.test_image_path = "./test06.jpg"
        self.output_dir = "./test_outputs"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def submit_multiple_jobs(self, num_jobs=5):
        """
        Submit multiple jobs concurrently to test async processing
        """
        session_ids = []
        
        # Submit multiple jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
            # Create a list of job submission tasks
            futures = [
                executor.submit(self.submit_single_job, 
                                initial_threshold=150 + i*10, 
                                invert=bool(i % 2)) 
                for i in range(num_jobs)
            ]
            
            # Collect session IDs
            for future in concurrent.futures.as_completed(futures):
                try:
                    session_id = future.result()
                    if session_id:
                        session_ids.append(session_id)
                except Exception as e:
                    print(f"Job submission error: {e}")
        
        return session_ids

    def submit_single_job(self, initial_threshold=150, invert=False):
        """
        Submit a single image processing job
        """
        try:
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

            response = requests.post(
                f"{self.base_url}/process", 
                json=json_data,
                headers={"Content-Type": "application/json"}
            )
            # # Prepare request data
            # with open(self.test_image_path, "rb") as image_file:
            #     files = {"image": image_file}
            #     data = {
            #         "max_iterations": "5",
            #         "initial_threshold": str(initial_threshold),
            #         "invert": str(invert).lower()
            #     }


            #     # Submit job
            #     response = requests.post(f"{self.base_url}/process", files=files, data=data)
            
            if response.status_code != 200:
                print(f"Job submission failed: {response.status_code}")
                return None

            # Extract session ID
            result = response.json()
            return result.get('session_id')

        except Exception as e:
            print(f"Error submitting job: {e}")
            return None

    def test_concurrent_job_submission(self, num_jobs=5):
        """
        Test concurrent job submission and processing
        """
        print(f"\nTesting concurrent submission of {num_jobs} jobs")
        
        # Submit multiple jobs
        session_ids = self.submit_multiple_jobs(num_jobs)
        
        # Verify all jobs were submitted
        assert len(session_ids) == num_jobs, f"Expected {num_jobs} jobs, got {len(session_ids)}"
        
        print("Job submission verified. Session IDs:", session_ids)
        return session_ids

    def test_job_processing_times(self, session_ids):
        """
        Verify jobs are processed asynchronously by checking processing times
        """
        print("\nTesting job processing times")
        
        # Track job processing times
        job_start_times = {}
        job_end_times = {}
        
        # Check status of each job
        for session_id in session_ids:
            job_start_times[session_id] = time.time()
        
        # Wait and check job statuses
        max_wait_time = 60  # 1 minute timeout
        start_overall_wait = time.time()
        
        # Track job completion
        completed_jobs = set()
        
        while len(completed_jobs) < len(session_ids):
            # Check overall timeout
            if time.time() - start_overall_wait > max_wait_time:
                raise TimeoutError("Jobs did not complete within expected time")
            
            # Check each job's status
            for session_id in session_ids:
                if session_id in completed_jobs:
                    continue
                
                try:
                    # Check job status
                    response = requests.get(f"{self.base_url}/status/{session_id}")
                    
                    if response.status_code != 200:
                        print(f"Failed to get status for {session_id}")
                        continue
                    
                    status_data = response.json()
                    status = status_data.get('job_info', {}).get('status')
                    
                    # Mark job as completed
                    if status == 'completed':
                        job_end_times[session_id] = time.time()
                        completed_jobs.add(session_id)
                        print(f"Job {session_id} completed")
                
                except Exception as e:
                    print(f"Error checking job status: {e}")
            
            # Wait before next status check
            time.sleep(1)
        
        # Analyze processing times
        processing_times = {}
        for session_id in session_ids:
            processing_times[session_id] = job_end_times[session_id] - job_start_times[session_id]
        
        print("\nJob Processing Times:")
        for session_id, proc_time in processing_times.items():
            print(f"Job {session_id}: {proc_time:.2f} seconds")
        
        # Verify jobs were processed concurrently
        max_time_diff = max(processing_times.values()) - min(processing_times.values())
        print(f"\nMax time difference between jobs: {max_time_diff:.2f} seconds")
        
        # Allow small time difference due to system load
        assert max_time_diff < 10, "Jobs were not processed concurrently"
        
        return processing_times

    def run_async_tests(self):
        """
        Run comprehensive async processing tests
        """
        try:
            # Test concurrent job submission
            session_ids = self.test_concurrent_job_submission()
            
            # Test job processing times
            self.test_job_processing_times(session_ids)
            
            print("\n--- All Async Tests Passed Successfully! ---")
            return True
        
        except AssertionError as ae:
            print(f"Async Test Failed: {ae}")
            return False
        except Exception as e:
            print(f"Unexpected error in async tests: {e}")
            return False

def main():
    # Create test instance
    test = AsyncImageProcessingTest()
    
    # Run async tests
    test.run_async_tests()

if __name__ == "__main__":
    main()