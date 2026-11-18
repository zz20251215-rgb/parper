class EarlyStopping:
    def __init__(self, min_epochs=30, patience=15, min_delta=0.002):
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('-inf')
        self.stale_epochs = 0

    def update(self, epoch, score):
        if score > self.best + self.min_delta:
            self.best = score
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return epoch >= self.min_epochs and self.stale_epochs >= self.patience
