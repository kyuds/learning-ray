import ray

# testing functions as parameters in ray framework.

class LocalExecutor:
    def __init__(self):
        self.func = None
    
    def setFunc(self, func):
        assert self.func is None
        self.func = func
        return self
    
    def execute(self):
        return self.func()

@ray.remote
class RemoteExecutor:
    def __init__(self, executor):
        self.executor = executor

    def exec(self):
        return self.executor.execute()

if __name__ == "__main__":
    executors = [
        RemoteExecutor.remote(LocalExecutor().setFunc(lambda: i))
        for i in range(5)
    ]
    results = [e.exec.remote() for e in executors]
    for r in ray.get(results):
        print(r)
