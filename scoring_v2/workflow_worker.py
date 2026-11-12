"""Container entry point. Never import or execute this worker on the host."""
import contextlib
import io
import json
import math
import sys

LIMIT = 1_048_576


class Capture(io.TextIOBase):
    def __init__(self):
        self.text = ''

    def write(self, value):
        self.text += value[:max(0, 4096 - len(self.text))]
        return len(value)


def native(value, depth=0):
    if depth > 32:
        return False
    if value is None or type(value) in (str, bool):
        return True
    if type(value) is int:
        return abs(value) <= 10**300
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(native(v, depth + 1) for v in value)
    return type(value) is dict and all(type(k) is str and native(v, depth + 1) for k, v in value.items())


def main():
    stdout, stderr = Capture(), Capture()
    try:
        request = json.loads(sys.stdin.buffer.read(2 * LIMIT + 1))
        namespace = {'__name__': '__submission__'}
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(request['code'], '<submission>', 'exec'), namespace)
            result = namespace['solve'](request['inputs'])
        if type(result) is not dict or not native(result):
            raise ValueError('Return a finite JSON-native dictionary')
        encoded = json.dumps(result, allow_nan=False)
        if len(encoded.encode()) > LIMIT // 2:
            raise ValueError('Returned output exceeds size limit')
        envelope = {'ok': True, 'outputs': result}
    except BaseException as exc:
        envelope = {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)[:500]}
    envelope.update(stdout=stdout.text, stderr=stderr.text)
    sys.stdout.write(json.dumps(envelope, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
