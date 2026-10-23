import hashlib

class ExactCache:
    def __init__(self):
        self.store = {}
        self.hits = 0
        self.misses = 0

    def make_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str):
        key = self.make_key(prompt)
        if(key in self.store):
            self.hits +=1
            return self.store[key]
        else:
            self.misses +=1
            return None
        # TODO 1: if key is in self.store, add 1 to self.hits
        #         and return the stored answer
        # TODO 2: otherwise add 1 to self.misses and return None

    def set(self, prompt: str, answer: str):
        key = self.make_key(prompt)
        self.store[key] = answer
        # TODO 3: save the answer in self.store under this key


cache = ExactCache()
print(cache.get("hello"))
cache.set("hello", "Hi there!")
print(cache.get("hello"))
print(cache.get("Hello"))
print(cache.hits, cache.misses)

print(cache.make_key("hello"))
print(cache.make_key("Hello"))