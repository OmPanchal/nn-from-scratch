class ActivationBlock:
    def __init__(self, activations=None):
        self.input = None

        if activations is not None:
            assert len(list(activations)) > 1, "There should be more than one activations"
            self.activations: list = activations

        else: self.activations = None

    def forward(self, input):
        self.input = input

        for activation in self.activations:
            self.input = activation.forward(self.input)

        return self.input

    def backwards(self, error, learning_rate):
        self.error = error

        for activation in reversed(self.activations):
            self.error = activation.backwards(self.error, learning_rate)

        return self.error
