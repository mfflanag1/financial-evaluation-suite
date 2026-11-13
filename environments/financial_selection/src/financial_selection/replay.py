"""Execute submitted source only in a fresh, public-data container.

The privilege-dropping controller comes from the pinned financial-research
runtime. The private verifier and population are never staged in this runtime.
Infrastructure failure raises, and never becomes a candidate financial zero.
"""
import asyncio
import base64
import hashlib
import json
import math

from .contract import PacketError, input_digest, parse_public, strict_json

MAX_SOURCE_BYTES = 1024 * 1024


class ReplayInfrastructureError(RuntimeError):
    pass


def source_valid(source):
    if type(source) is not bytes or not 0 < len(source) <= MAX_SOURCE_BYTES:
        return False
    try:
        return bool(source.decode('utf-8').strip())
    except UnicodeError:
        return False


async def run_public_program(public_files, source, *, runtime_config, timeout_seconds=30):
    """Return candidate status and private captured output; no scoring performed."""
    parse_public(public_files)
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 120):
        raise ReplayInfrastructureError('invalid_timeout')
    if not source_valid(source):
        return {'status': 'candidate_failure', 'reason': 'invalid_source', 'policy_bytes': None}
    from financial_research.reproduction import _CONTROLLER, fresh_config
    from verifiers.v1.runtimes import provision_runtime
    try:
        config = fresh_config(runtime_config)
        async with provision_runtime(config, env={}) as runtime:
            await runtime.backend.prepare_setup()
            await runtime.write('/opt/selection-replay/controller.py', _CONTROLLER.encode())
            setup = await runtime.run(['chmod', '700', '/opt/selection-replay'], {})
            if setup.exit_code:
                raise ReplayInfrastructureError('controller_setup_failed')
            for name, content in public_files.items():
                await runtime.write('/workspace/' + name, content.encode())
            await runtime.write('/workspace/research.py', source)
            await runtime.backend.prepare_execution([])
            async with asyncio.timeout(timeout_seconds + 15):
                completed = await runtime.run(['python', '-I', '/opt/selection-replay/controller.py',
                                               str(timeout_seconds)], {})
            if completed.exit_code:
                raise ReplayInfrastructureError('controller_failed')
            captured = json.loads(await runtime.read('/opt/selection-replay/result.json', max_bytes=3 * 1024 * 1024))
            if captured.get('reason') != 'output_created':
                allowed = {'analysis_exit', 'analysis_timeout', 'output_not_regular',
                           'output_too_large', 'output_missing_or_unsafe'}
                if captured.get('reason') not in allowed:
                    raise ReplayInfrastructureError('invalid_controller_result')
                return {'status': 'candidate_failure', 'reason': captured['reason'], 'policy_bytes': None}
            try:
                payload = base64.b64decode(captured['policy'], validate=True)
                if not 0 < len(payload) <= 2 * 1024 * 1024:
                    raise ValueError('output size')
                strict_json(payload.decode('utf-8'))
            except (PacketError, ValueError, UnicodeError, KeyError, RecursionError):
                return {'status': 'candidate_failure', 'reason': 'invalid_output_json', 'policy_bytes': None}
            return {'status': 'passed', 'reason': 'output_created', 'policy_bytes': payload,
                    'source_sha256': hashlib.sha256(source).hexdigest()}
    except ReplayInfrastructureError:
        raise
    except Exception:
        raise ReplayInfrastructureError('replay_service_failed') from None


async def verify_program(public_files, source, submitted, fresh_packets, *, runtime_config, timeout_seconds=30):
    """Bind same-input reproduction, then average exact scores on fresh packets.

    Only trusted callers choose fresh packets. Reproduction failure zeros the
    aggregate. Individual fresh-input candidate failures score zero; service
    failures abort the record. This verifies this source on these packets, not
    unrestricted generalization or model capability.
    """
    from .verifier import grade
    if type(fresh_packets) is not list or not 1 <= len(fresh_packets) <= 4:
        raise ReplayInfrastructureError('invalid_fresh_packet_count')
    for packet in [public_files, *fresh_packets]:
        parse_public(packet)
    identities = [input_digest(packet) for packet in [public_files, *fresh_packets]]
    if len(set(identities)) != len(identities):
        raise ReplayInfrastructureError('fresh_inputs_are_not_distinct')
    first_grade = grade(public_files, submitted)
    if first_grade['status'] != 'scored':
        return {'status': 'candidate_failure', 'reason': 'invalid_submitted_policy',
                'reward': 0.0, 'reproduction_verified': False, 'packet_rewards': []}
    reproduced = await run_public_program(public_files, source, runtime_config=runtime_config,
                                          timeout_seconds=timeout_seconds)
    try:
        matches = (reproduced['status'] == 'passed'
                   and grade(public_files, reproduced['policy_bytes'])['status'] == 'scored'
                   and strict_json(reproduced['policy_bytes'].decode()) == strict_json(submitted.decode() if isinstance(submitted, bytes) else submitted))
    except (ValueError, UnicodeError, TypeError):
        matches = False
    if not matches:
        return {'status': 'candidate_failure', 'reason': 'policy_not_reproduced',
                'reward': 0.0, 'reproduction_verified': False, 'packet_rewards': []}
    rewards = [first_grade['reward']]
    for packet in fresh_packets:
        result = await run_public_program(packet, source, runtime_config=runtime_config,
                                          timeout_seconds=timeout_seconds)
        rewards.append(grade(packet, result['policy_bytes'])['reward'] if result['status'] == 'passed' else 0.0)
    return {'status': 'scored', 'reason': 'program_checked', 'reward': sum(rewards) / len(rewards),
            'reproduction_verified': True, 'packet_rewards': rewards,
            'source_sha256': hashlib.sha256(source).hexdigest()}
