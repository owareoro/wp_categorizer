class RunningStats:
    def __init__(self):
        self._n = 0
        self._mean = 0
        self._M2 = 0
        self._variance = 0

    def add(self, p):
        self._n += 1
        delta = p - self._mean
        self._mean += delta / self._n
        delta2 = p - self._mean
        self._M2 += delta * delta2

        if self._n > 1:
            self._variance = self._M2 / (self._n - 1)
        return self
