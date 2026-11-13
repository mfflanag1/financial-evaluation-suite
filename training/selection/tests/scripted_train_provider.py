"""Synthetic token server and local tokenizer; no model weights or inference."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import numpy as np


def make_tokenizer(directory):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    core = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(alphabet)}, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core,
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>",
                                   "<tool_call>", "</tool_call>", "<tool_response>",
                                   "</tool_response>", "<think>", "</think>"],
        eos_token="<|im_end|>", model_max_length=262144)
    tokenizer.save_pretrained(directory)
    return tokenizer


def tool(command):
    return '<tool_call>\n' + json.dumps({"name": "bash", "arguments": {"command": command}}) + '\n</tool_call><|im_end|>'


class ScriptedTrainProvider:
    """Actual generate wire protocol with explicitly synthetic token evidence.

    Its deterministic fixture outputs and logprobs are protocol tests, never
    measurements of policy quality or evidence of optimization.
    """
    def __init__(self, tokenizer, responses):
        self.tokenizer, self.responses = tokenizer, list(responses)
        self.calls, self.completions = [], []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                payload = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.reply(200, {"data": [{"id": "fixture-policy", "max_model_len": 262144}]})
                else:
                    self.reply(404, {"error": "unsupported fixture path"})

            def do_POST(self):
                if self.path != "/inference/v1/generate":
                    self.reply(404, {"error": "unsupported fixture path"})
                    return
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                index = len(fixture.calls)
                fixture.calls.append(request)
                if index >= len(fixture.responses):
                    self.reply(400, {"error": "fixture response schedule exhausted"})
                    return
                ids = fixture.tokenizer.encode(fixture.responses[index], add_special_tokens=False)
                fixture.completions.append(ids)
                probabilities = np.full(len(ids), -0.25, dtype=np.float32)
                packed = {"data": base64.b64encode(probabilities.tobytes()).decode(),
                          "shape": [len(ids)], "dtype": "float32"}
                count = len(request["token_ids"])
                self.reply(200, {"request_id": f"fixture-{index}", "choices": [{
                    "token_ids": ids, "finish_reason": "stop", "completion_logprobs": packed,
                    "sampling_mask": [[token] for token in ids],
                    "weight_versions": [{"version": "fixture-policy-v1", "start": 0, "end": len(ids)}]}],
                    "usage": {"prompt_tokens": count, "completion_tokens": len(ids),
                              "total_tokens": count + len(ids)}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
