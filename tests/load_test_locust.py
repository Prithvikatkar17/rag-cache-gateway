import random
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