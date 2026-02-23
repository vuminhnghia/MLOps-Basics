import random
from locust import HttpUser, task, between

INPUTS = [
    "The boy is sitting on a bench",
    "This sentence it not make sense",
    "The cat sat on the mat",
    "Him go store yesterday for milk",
    "She reads books every evening",
    "The children are playing in the park",
    "Me want eat pizza now please",
    "Scientists discovered a new species of fish",
    "They goes to school every day",
    "The government announced a new policy on climate change",
]


class InferenceUser(HttpUser):
    # Simulate realistic users: wait 1-3s between requests
    wait_time = between(1, 3)

    @task(9)
    def predict(self):
        text = random.choice(INPUTS)
        self.client.get("/predict", params={"text": text}, name="/predict")

    @task(1)
    def health_check(self):
        self.client.get("/", name="/")
