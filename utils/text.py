import time


class Severity:
    def __init__(self, low, high, name):
        self.low = low
        self.high = high
        self.name = name


class TextFlag:
    INFO = Severity('\033[93m', '\033[0m', 'INFO')
    WARNING = Severity('\033[92m', '\033[0m', 'WARNING')
    ERROR = Severity('\033[91m', '\033[0m', 'ERROR')


def log(severity: Severity, text):
    print(severity.low, f'{time.time():.2f}', '\t', f'{severity.name}', '\t', text, severity.high)
