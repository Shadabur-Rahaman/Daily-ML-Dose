class EarlyStopping:
    def __init__(self, patience=3):
        self.best_loss = float('inf')
        self.counter = 0
        self.patience = patience

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
