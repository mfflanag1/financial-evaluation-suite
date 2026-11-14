"""Collector control tests with labelled native API doubles; no GPU execution claim."""
import asyncio
import ast
import copy
from dataclasses import dataclass
import hashlib
from importlib.resources import files
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src/financial_selection_supervised"
if importlib.util.find_spec("financial_selection_supervised") is None:
    sys.path.insert(0, str(SOURCE.parent))
from financial_selection_supervised import gpu_observer as observer


class TensorDouble:
    """Explicit CPU API double, never a native tensor or evidence of CUDA."""
    def __init__(self, values, dtype="fixture.float64"):
        self.values = list(values)
        self.dtype = dtype
        self.shape = (len(values),)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def reshape(self, *_): return self
    def view(self, *_): return self
    def numpy(self): return self
    def tobytes(self): return b"".join(struct.pack("d", value) for value in self.values)
    def tolist(self): return self.values.copy()
    def numel(self): return len(self.values)
    def double(self): return self
    def square(self): return TensorDouble([value * value for value in self.values])
    def sum(self): return TensorDouble([sum(self.values)])
    def all(self): return TensorDouble([all(self.values)])
    def item(self):
        assert len(self.values) == 1
        return self.values[0]


TORCH_DOUBLE = types.SimpleNamespace(uint8="fixture.uint8", isfinite=lambda value:
                                    TensorDouble([int(math.isfinite(x)) for x in value.values]),
                                    count_nonzero=lambda value: TensorDouble([sum(x != 0 for x in value.values)]))


@dataclass
class ProgressDouble:
    step: int = 1
    total_tokens: int = 0
    total_samples: int = 0


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def store(self):
        root = self.root / "evidence"
        root.mkdir()
        config = root / "config.json"
        sources = root / "sources.json"
        config.write_text('{}')
        sources.write_text('{}')
        plan = {"world_size": 1, "step": 1, "update_scope": "lora", "run_id": "fixture-only",
                "collector_sha256": observer.file_sha(observer.__file__),
                "config_sha256": observer.file_sha(config), "sources_sha256": observer.file_sha(sources)}
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan))
        return observer.Store(path, root)

    def test_json_inventory_preserves_empty_containers_and_pointer_names(self):
        value = {"a/b": {"~": 1}, "a": {"b": 2}, "empty": [], "none": None}
        result = observer.flatten(value)
        self.assertEqual(set(result), {"a~1b/~0", "a/b", "empty", "none"})
        self.assertNotEqual(result["a~1b/~0"], result["a/b"])

    def test_nonfinite_metadata_and_opaque_objects_rejected(self):
        for value in (float("nan"), float("inf"), object()):
            with self.subTest(value=type(value).__name__), self.assertRaises((RuntimeError, TypeError, ValueError)):
                observer.flatten({"value": value})

    def test_tensor_hash_observes_dtype_shape_and_bytes(self):
        with patch.dict(sys.modules, {"torch": TORCH_DOUBLE}):
            first = observer.tensor_sha(TensorDouble([1., 2.]))
            self.assertNotEqual(first, observer.tensor_sha(TensorDouble([1., 3.])))
            self.assertNotEqual(first, observer.tensor_sha(TensorDouble([1., 2.], dtype="other")))

    def test_patch_scope_restores_reverse_order_and_rejects_other_owner(self):
        owner = types.SimpleNamespace(value=1)
        scope = observer.PatchScope()
        scope.patch(owner, "value", 2)
        scope.patch(owner, "value", 3)
        scope.close()
        self.assertEqual(owner.value, 1)
        scope.patch(owner, "value", 4)
        owner.value = 5
        with self.assertRaisesRegex(RuntimeError, "Another owner"):
            scope.close()
        self.assertEqual(owner.value, 5)

    def test_store_refuses_overwrite_traversal_symlink_and_source_drift(self):
        store = self.store()
        store.write("raw/a.json", {"value": 1})
        with self.assertRaises(FileExistsError):
            store.write("raw/a.json", {"value": 2})
        with self.assertRaises(RuntimeError):
            store.write("../escape.json", {})
        outside = self.root / "outside"
        outside.mkdir()
        (store.root / "link").symlink_to(outside)
        with self.assertRaises(RuntimeError):
            store.write("link/escape.json", {})
        (store.root / "sources.json").write_text('{"drift":true}')
        with self.assertRaisesRegex(RuntimeError, "artifact binding"):
            observer.Store(self.root / "plan.json", store.root)

    def test_broadcast_copy_excludes_mutating_handshake_markers(self):
        store = self.store()
        source = self.root / "broadcast"
        source.mkdir()
        for name in (".sender_ready", ".receiver_ready", ".started", ".finished", "adapter_config.json", "adapter_model.safetensors"):
            (source / name).write_bytes(name.encode())
        result = observer._copy_files(store, source, "published/step_1", broadcast=True)
        self.assertEqual(set(result), {"published/step_1/adapter_config.json", "published/step_1/adapter_model.safetensors"})

    def test_model_manifest_binds_all_files_and_rejects_symlinks(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text('{}')
        (model / "model.safetensors").write_bytes(b"fixture only, not model weights")
        first = observer.model_manifest(model, "fixture", "revision")
        (model / "tokenizer.json").write_text('{}')
        second = observer.model_manifest(model, "fixture", "revision")
        self.assertNotEqual(observer.digest(first), observer.digest(second))
        (model / "alias").symlink_to(model / "config.json")
        with self.assertRaisesRegex(RuntimeError, "Materialize"):
            observer.model_manifest(model, "fixture", "revision")

    @staticmethod
    def response(ids=(1, 2), probabilities=(-.2, -.3), version="0", request_id="native-fixture-id"):
        tokens = types.SimpleNamespace(completion_ids=list(ids), completion_logprobs=list(probabilities),
                 weight_versions=[types.SimpleNamespace(start=0, end=len(ids), version=version)])
        return types.SimpleNamespace(tokens=tokens, id=request_id, model="fixture-policy")

    def test_response_observation_preserves_native_values_and_request_identity(self):
        response = self.response()
        body = {"messages": [{"role": "user", "content": "fixture"}]}
        row = observer.response_observation(response, body, "trace-fixture")
        self.assertEqual(row["provider_request_id"], response.id)
        self.assertEqual(row["logprobs"], response.tokens.completion_logprobs)
        self.assertEqual(row["weight_versions"], [0, 0])
        self.assertEqual(row["prompt_sha256"], observer.digest(body))

    def test_response_observation_rejects_missing_id_bad_tokens_and_versions(self):
        for response in (self.response(request_id=""), self.response(ids=(True, 2)),
                         self.response(probabilities=(float("nan"), -.1)),
                         self.response(probabilities=(.1, -.2)), self.response(version="unknown")):
            with self.assertRaises(RuntimeError):
                observer.response_observation(response, {}, "trace")
        response = self.response()
        response.tokens.weight_versions[0].start = 1
        with self.assertRaises(RuntimeError):
            observer.response_observation(response, {}, "trace")

    def test_gradient_observation_checks_every_trainable_element(self):
        parameter = types.SimpleNamespace(requires_grad=True, grad=TensorDouble([3., 4.]))
        model = types.SimpleNamespace(named_parameters=lambda: [("adapter", parameter)])
        with patch.dict(sys.modules, {"torch": TORCH_DOUBLE}):
            self.assertEqual(observer.gradient_observation(model),
                             {"l2_norm": 5., "total_elements": 2, "finite_elements": 2})
            parameter.grad = TensorDouble([3., float("nan")])
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                observer.gradient_observation(model)
            parameter.grad = None
            with self.assertRaisesRegex(RuntimeError, "no observed gradient"):
                observer.gradient_observation(model)

    def test_pre_state_observation_does_not_call_appstate_or_step(self):
        parameter = type("ParameterDouble", (), {"requires_grad": True})()
        model = types.SimpleNamespace(named_parameters=lambda: [("adapter", parameter)])
        optimizer = types.SimpleNamespace(state={}, state_dict=lambda: {"state": {}, "param_groups": [{"lr": 1e-6}]})
        scheduler = types.SimpleNamespace(state_dict=lambda: {"last_epoch": 0})
        def app_state(*_):
            self.fail("Initial observation must not invoke DCP AppState lazy optimizer initialization")
        native = types.SimpleNamespace(AppState=app_state, get_model_state_dict=lambda _: {"adapter": 0})
        with patch.dict(sys.modules, {"prime_rl.trainer.ckpt": native}):
            result = observer.snapshot(model, optimizer, scheduler, ProgressDouble(), initialized=False)
        self.assertEqual(result["optimizer_steps"], {"adapter": 0})
        self.assertEqual(result["state"]["progress"]["step"], observer.digest(1))

    def test_saved_adapter_canonicalization_requires_named_native_bytes(self):
        with patch.dict(sys.modules, {"torch": TORCH_DOUBLE}):
            tensor = TensorDouble([1., 2.])
            checksum = observer.tensor_sha(tensor)
            full = {"model.layers.0.q_proj.lora_A": checksum}
            result = observer.canonical_adapter_inventory({"model.layers.0.q_proj.lora_A.weight": tensor}, full)
            self.assertEqual(result, full)
            with self.assertRaises(RuntimeError):
                observer.canonical_adapter_inventory({"unmapped.lora_A.weight": tensor}, full)
            with self.assertRaises(RuntimeError):
                observer.canonical_adapter_inventory({"model.layers.0.q_proj.lora_A.weight": TensorDouble([9.])}, full)

    def test_batch_join_requires_exact_cohort_token_stream(self):
        store = self.store()
        (store.root / "cohorts").mkdir()
        members = [{"trace_id": "trace-1", "token_ids": [10], "logprobs": [-.5], "token_advantages": [.2]}]
        row = {"cohort_id": "fixture-cohort", "members": members}
        (store.root / "cohorts/x.json").write_text(json.dumps(row))
        batch = {"trace_ids": ["trace-1"], "sequence_lengths": [2],
                 "input_ids": TensorDouble([9, 10]), "loss_mask": TensorDouble([False, True]),
                 "inference_logprobs": TensorDouble([0., -.5]), "advantages": TensorDouble([0., .2])}
        result = observer._batch_observation(store, [batch])
        self.assertEqual(result["batch_sampled_tokens"], 1)
        self.assertEqual(result["batch_micro_batches"], 1)
        self.assertEqual(result["batch_sha256"], observer.digest(members))
        batch["advantages"] = TensorDouble([0., .3])
        with self.assertRaisesRegex(RuntimeError, "differs"):
            observer._batch_observation(store, [batch])

    def test_missing_checkpoint_components_or_stages_cannot_be_reported_successful(self):
        store = self.store()
        checkpoint = self.root / "step_1"
        checkpoint.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            observer.finalize_checkpoint(store, checkpoint)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            observer.assemble_bundle(store)
        self.assertFalse((store.root / "bundle.json").exists())

    def test_native_load_initialization_records_zero_update_and_rejects_second_call(self):
        parameter = TensorDouble([1., 2.])
        parameter.requires_grad = True
        parameter.grad = TensorDouble([0., 0.])
        model = types.SimpleNamespace(named_parameters=lambda: [("adapter", parameter)])
        optimizer = types.SimpleNamespace(state={}, param_groups=[{"lr": 0.0, "params": [parameter]}])
        def native_step(closure=None):
            self.assertIsNone(closure)
            optimizer.state[parameter] = {"step": 1}
            return "native return"
        optimizer.step = native_step
        with patch.dict(sys.modules, {"torch": TORCH_DOUBLE}):
            witnesses = observer.guard_native_optimizer_initialization(model, optimizer)
            self.assertEqual(optimizer.step(closure=None), "native return")
            self.assertEqual(len(witnesses), 1)
            self.assertEqual(witnesses[0]["model_parameters_before"], witnesses[0]["model_parameters_after"])
            with self.assertRaises(RuntimeError):
                optimizer.step(closure=None)

    def test_native_load_guard_rejects_nonzero_rate_grad_closure_and_parameter_change(self):
        for defect in ("rate", "gradient", "nonfinite", "closure", "change", "missing_parameter"):
            with self.subTest(defect=defect):
                parameter = TensorDouble([1.])
                parameter.requires_grad = True
                parameter.grad = TensorDouble([1. if defect == "gradient" else float("nan") if defect == "nonfinite" else 0.])
                model = types.SimpleNamespace(named_parameters=lambda: [("adapter", parameter)])
                optimizer = types.SimpleNamespace(state={}, param_groups=[{
                    "lr": 1e-6 if defect == "rate" else 0.,
                    "params": [] if defect == "missing_parameter" else [parameter]}])
                def native_step(closure=None):
                    if defect == "change":
                        parameter.values[0] = 2.
                    optimizer.state[parameter] = {"step": 1}
                optimizer.step = native_step
                with patch.dict(sys.modules, {"torch": TORCH_DOUBLE}):
                    witnesses = observer.guard_native_optimizer_initialization(model, optimizer)
                    with self.assertRaises(RuntimeError):
                        optimizer.step(closure=(lambda: None) if defect == "closure" else None)
                    self.assertEqual(witnesses, [])

    def test_real_cpu_scheduler_can_be_wrapped_after_native_initialization(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Native Torch is absent; this CPU scheduler regression did not run")
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        previous = optimizer.step
        observed = []
        def step(*args, **kwargs):
            observed.append(True)
            return previous(*args, **kwargs)
        scope = observer.PatchScope()
        scope.patch(optimizer, "step", step)
        parameter.sum().backward()
        optimizer.step()
        scheduler.step()
        scope.close()
        self.assertEqual(observed, [True])
        self.assertEqual(int(optimizer.state[parameter]["step"].item()), 1)

    def test_checkpoint_load_call_matches_exact_pinned_native_setup_model_signature(self):
        try:
            native_bytes = files("prime_rl").joinpath("trainer/model.py").read_bytes()
            pins = json.loads(files("financial_selection_training").joinpath("source_bindings.json").read_text())
        except ModuleNotFoundError:
            self.skipTest("Pinned native packages are absent; run this contract check in the training environment")
        self.assertEqual(hashlib.sha256(native_bytes).hexdigest(), pins["native_sources"]["prime_rl/trainer/model.py"])
        native = next(node for node in ast.parse(native_bytes).body
                      if isinstance(node, ast.FunctionDef) and node.name == "setup_model")
        # Keep the real pinned signature and replace only the GPU-executing body.
        bound_native = copy.deepcopy(native)
        bound_native.body = ast.parse("return locals()").body
        bound_native.decorator_list = []
        definition = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                      bound_native], type_ignores=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(definition), "pinned-native-signature", "exec"), namespace)
        observer_tree = ast.parse(Path(observer.__file__).read_text())
        loader = next(node for node in observer_tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "checkpoint_load")
        calls = [node for node in ast.walk(loader) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "setup_model"]
        self.assertEqual(len(calls), 1)
        config, dimensions = object(), object()
        namespace.update(config=types.SimpleNamespace(model=config), dims=dimensions)
        actual_call = ast.Expression(body=copy.deepcopy(calls[0]))
        bound = eval(compile(ast.fix_missing_locations(actual_call), "actual-reload-call", "eval"), namespace)
        self.assertIs(bound["config"], config)
        self.assertIs(bound["parallel_dims"], dimensions)
        self.assertIs(bound["loading_from_checkpoint_later"], False)
        self.assertIs(bound["init_from_scratch"], False)
        broken_call = copy.deepcopy(actual_call)
        for keyword in broken_call.body.keywords:
            if keyword.arg == "loading_from_checkpoint_later":
                keyword.arg = "loading_from_ckpt_later"
        with self.assertRaisesRegex(TypeError, "loading_from_ckpt_later"):
            eval(compile(ast.fix_missing_locations(broken_call), "regression-wrong-keyword", "eval"), namespace)


if __name__ == "__main__":
    unittest.main()
