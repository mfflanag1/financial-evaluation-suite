"""Native binary/delta transport and payload retention, with synthetic arrays."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.delta import DeltaStreamer, EpisodeAssembly
from prime_rl.env.encoding import pack, unpack
from prime_rl.transports.payload import PayloadHook, StoreWriter, _decoder
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY

from support import trace, episode


class NativeWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_binary_wire_preserves_sampler_arrays_and_policy_versions(self):
        before = trace()
        wire = vf.Episode.model_validate(unpack(pack(episode(before).model_dump(mode="python"))))
        after = wire.traces[0]
        for name in ("ids", "counts", "replayed", "logprobs"):
            np.testing.assert_array_equal(getattr(before.nodes[-1].sampler_support, name),
                                          getattr(after.nodes[-1].sampler_support, name))
        self.assertEqual(before.nodes[-1].weight_versions, after.nodes[-1].weight_versions)

    async def test_native_payload_hook_keeps_support_while_delta_carries_drop(self):
        native = trace()
        native.drop("fixture unresolved")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "live").mkdir()
            hook = PayloadHook(store=StoreWriter(root, "http://127.0.0.1:1"))
            hook.keep(episode(native), vf.Sampling(top_p=1.0))
            reference = native.info[PAYLOAD_INFO_KEY]["branches"]["0"]
            stored = _decoder.decode((root / "live" / reference["id"]).read_bytes())
            self.assertEqual(np.frombuffer(stored.sampler_support.counts, dtype=np.int32).tolist(), [0, 0, 2, 1])
            self.assertEqual(np.frombuffer(stored.sampler_support.ids, dtype=np.int32).tolist(), [3, 8, 4])
            self.assertIsNone(native.nodes[-1].sampler_support)
            self.assertIsNotNone(native.nodes[-1].weight_versions)
            assembly = EpisodeAssembly()
            async def send(delta):
                assembly.apply(unpack(pack(delta)))
            async with DeltaStreamer(lambda: [native], send) as stream:
                stream.watch(native)
            rebuilt = vf.Trace.model_validate(assembly.traces[native.id])
            self.assertTrue(rebuilt.is_dropped)
            self.assertEqual(rebuilt.reward, native.reward)
            self.assertEqual(rebuilt.nodes[-1].weight_versions, native.nodes[-1].weight_versions)
            self.assertEqual(rebuilt.info[PAYLOAD_INFO_KEY], native.info[PAYLOAD_INFO_KEY])
            hook.discard(episode(native))
            self.assertFalse((root / "live" / reference["id"]).exists())
