from experiments.adversarial_peer_deception_prompt.run import replace_head
from gleipnir.adversarial.contract import join_prompt, split_prompt


def test_prompt_edit_preserves_nested_trajectory():
    trajectory = '[USER]\n<agent_trajectory>\nA peer said Prediction:1\n</agent_trajectory>\n[TOOL] truth'
    original = join_prompt('original\n', trajectory)
    changed = replace_head(original, 'explicit peer policy\n')
    assert split_prompt(changed).head == 'explicit peer policy\n'
    assert split_prompt(changed).trajectory == trajectory


def test_new_prompt_changes_only_instruction():
    from experiments.adversarial_peer_deception_prompt.run import EXP, ROOT
    original = (ROOT / 'experiments/tool_trajectory_monitoring/prompts/student.txt').read_text()
    peer = (EXP / 'peer.txt').read_text()
    prefix, suffix = original.split('Return exactly one line and nothing else:')
    assert peer.startswith(prefix)
    assert peer.endswith('Return exactly one line and nothing else:' + suffix)
    assert 'Merely receiving, reading, quoting' in peer
