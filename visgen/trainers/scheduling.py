class ExponentialScheduler:
    def __init__(self, initial=0.1, decay_rate=15):
        self.initial = initial
        self.decay_rate = decay_rate

    def get_lr(self, epoch):
        return self.initial * (0.1 ** (epoch // self.decay_rate))

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.get_lr(epoch)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class PolynomialScheduler:
    def __init__(self, initial=0.1, max_epoch=100, decay_rate=15):
        self.initial = initial
        self.decay_rate = decay_rate
        self.max_epoch = max_epoch

    def get_lr(self, epoch):
        return self.initial * ((1 - float(epoch) / self.max_epoch) ** self.decay_rate)

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr
