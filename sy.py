import os

class StableDiffusionPipeline():
    def __init__(self):
        pass

    @property
    def output(self):
        return 3

    def calculate(self):
        return self.output + self.output

test_object = StableDiffusionPipeline()
output = test_object.calculate()
print(output)