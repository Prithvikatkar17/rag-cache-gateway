import random
import uuid
from locust import HttpUser, task, between

PROMPTS = [
    "How do I reset my password?",
    "Is my data safe?",
    "How many days do I have to ask for a refund?",
    "What plans does AcmeCloud offer?",
    "How do I delete my account?",
]

class QueryUser(HttpUser):
    # TODO 1: Tell each simulated user to wait between 0.5 and 2 seconds 
    # before firing their next request. This mimics real human behavior.
    wait_time = between(0.5, 2)

    # The @task decorator tells Locust: "This is the action the user should do repeatedly."
    @task
    def ask(self):
        # TODO 2: Pick a random question from our list for this request
        prompt = random.choice(PROMPTS)
        
        # catch_response=True lets us manually grade if the response was a success or failure
        with self.client.post("/query", json={"prompt": prompt},
                              catch_response=True, name="/query") as resp:
            
            # TODO 3: Grade the response
            # First, check if the server crashed or threw an error (like a 404 or 500)
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            else:
                data = resp.json()
                # Check if the API returned a valid cache status. 
                # (Including EXACT and SEMANTIC hits from our previous code!)
                valid_statuses = ["CACHE_HIT", "CACHE_MISS", "EXACT_CACHE_HIT", "SEMANTIC_CACHE_HIT"]
                
                if data.get("status") not in valid_statuses:
                    resp.failure("unexpected status")
                else:
                    # If the status code is 200 AND the JSON looks right, mark it as a pass
                    resp.success()

class MissUser(HttpUser):
    # Each simulated user pauses between 0.5 and 2 seconds between requests
    wait_time = between(0.5, 2)

    @task
    def ask(self):
        # TODO 1: Create a completely random question using UUID
        # uuid4() generates a random 36-character string like "123e4567-e89b-12d3-a456-426614174000"
        prompt = f"Unique question {uuid.uuid4()}"
        
        # We use name="/query" so Locust groups all these random URLs under one clean label in the dashboard
        with self.client.post("/query", json={"prompt": prompt},
            catch_response=True, name="/query") as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            server_ms = resp.json()["execution_time_ms"]
            client_ms = resp.elapsed.total_seconds() * 1000
            if client_ms > 3000:
                resp.failure(f"slow: client {client_ms:.0f} ms, server {server_ms:.0f} ms")
            else:
                resp.success()
            
            # # TODO 2: Grade the response just like we did in QueryUser
            # if resp.status_code != 200:
            #     # The server crashed or refused the connection
            #     resp.failure(f"HTTP {resp.status_code}")
            # else:
            #     data = resp.json()
            #     valid_statuses = ["CACHE_HIT", "CACHE_MISS", "EXACT_CACHE_HIT", "SEMANTIC_CACHE_HIT"]
                
            #     # Check if our server replied with the correct JSON format
            #     if data.get("status") not in valid_statuses:
            #         resp.failure("unexpected status")
            #     else:
            #         # Everything went perfectly!
            #         resp.success()