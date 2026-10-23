score = 3.14749503409
name = "password"

print(f"{name} : {score:.3f}")


names = ["HNSW" ,"prith" , "python" , "AI" , "rag"]
scores = [0.34 , 0.756 , 0.55 , 0.23 , 0.99]

good = [ s for s in scores if s > 0.50]
lookup = {n : s for n,s in zip( names , scores)}

ranked = sorted(lookup.items() , key= lambda pair : -pair[1])
print(ranked)
print(good)

def is_hit(similarity: float, threshold:float) -> bool :
    return similarity >= threshold

print(is_hit(0.9,0.8))

class Counter:
    def __init__(self ,name:str):
        self.name = name
        self.count = 0

    def add(self):
        self.count+=1

requests = Counter("requests")
requests.add()
requests.add()
print(requests.name , requests.count)


# try / except. Handle an error instead of crashing.

try:
    number = int("abc")
except ValueError:
    print("number not found")