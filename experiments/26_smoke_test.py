import time
import httpx

URL = "http://127.0.0.1:8000/query"

def ask(client, prompt):
    # Start the stopwatch
    start = time.perf_counter()
    
    # TODO 1: Send the request to the server, check for errors, and parse the data
    response = client.post(URL, json={"prompt": prompt})
    response.raise_for_status()
    data = response.json()
    
    # Stop the stopwatch and convert to milliseconds
    client_ms = (time.perf_counter() - start) * 1000
    
    # TODO 2: Return a clean summary of what happened
    return {
        "status": data["status"],
        "server_ms": data["execution_time_ms"],
        "client_ms": round(client_ms, 2)
    }

def main():
    # A unique prompt using the current timestamp, guaranteeing the first call is a miss
    prompt = f"How do I securely access a remote server? ({int(time.time())})"
    
    # Open an HTTP session
    with httpx.Client(timeout=30) as client:
        # Ask the exact same question 3 times in a row
        for label in ["call 1", "call 2", "call 3"]:
            r = ask(client, prompt)
            print(f"{label}: {r['status']:<11} server {r['server_ms']:>9} ms | client {r['client_ms']:>9} ms")

        # Test what happens when we send a bad (empty) request
        bad = client.post(URL, json={})
        print("bad request ->", bad.status_code, bad.json()["detail"][0]["msg"])

if __name__ == "__main__":
    main()