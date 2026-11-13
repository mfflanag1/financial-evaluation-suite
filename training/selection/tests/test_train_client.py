"""Real CPU TrainClient renderer and HTTP wire, with no model weights."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.clients.train import TrainClient
from verifiers.v1.configs.client import TrainClientConfig
from verifiers.v1.dialects import ChatDialect

from scripted_train_provider import ScriptedTrainProvider, make_tokenizer, tool


class TrainClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_renderer_and_generate_protocol_preserve_training_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = make_tokenizer(Path(directory) / "tokenizer")
            expected = tool("printf fixture")
            with ScriptedTrainProvider(tokenizer, [expected]) as provider:
                client = TrainClient(TrainClientConfig(base_url=provider.url, api_key_var="FIXTURE_TOKEN",
                    renderer={"name": "qwen3"}, renderer_model_name=str(Path(directory) / "tokenizer")))
                try:
                    response = await client.get_response(ChatDialect(), {
                        "model": "fixture-policy", "messages": [
                            {"role": "system", "content": "Synthetic protocol fixture"},
                            {"role": "user", "content": "Run the fixture command"}],
                        "tools": [{"type": "function", "function": {"name": "bash", "description": "Shell",
                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                           "required": ["command"]}}}]},
                        vf.Sampling(top_p=1, max_tokens=4096), session_id="fixture-session")
                finally:
                    await client.close()
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(response.finish_reason, "tool_calls")
                self.assertEqual(response.message.tool_calls[0].name, "bash")
                np.testing.assert_array_equal(response.tokens.completion_ids, provider.completions[0])
                np.testing.assert_array_equal(response.tokens.completion_logprobs,
                                              np.full(len(provider.completions[0]), -0.25, dtype=np.float32))
                np.testing.assert_array_equal(response.tokens.sampler_support.ids, provider.completions[0])
                self.assertTrue(np.all(response.tokens.sampler_support.replayed))
                self.assertEqual(response.tokens.weight_versions[0].version, "fixture-policy-v1")
                self.assertEqual(response.tokens.weight_versions[0].end, len(provider.completions[0]))
