import model.layer as L


class SGDOptimizer:
    def __init__(self, lr, ld=0, decay_rate=0, decay_step=1000):
        self.lr = lr
        self.ld = ld
        self.decay_rate = decay_rate if 0 < decay_rate < 1 else None
        self.decay_step = decay_step if 0 < decay_rate < 1 else None
        self.iteration = 0

    def step(self, model):
        for layer in model.layers:
            if isinstance(layer, L.Linear):
                layer.w -= self.lr * (layer.dw + self.ld * layer.w)
                layer.b -= self.lr * layer.db
                layer.zero_grad()
            self.iteration += 1
            # lr decay per decay_step steps
            if self.decay_rate and self.iteration % self.decay_step == 0:
                self.lr *= self.decay_rate if self.lr > 0.01 else 1
