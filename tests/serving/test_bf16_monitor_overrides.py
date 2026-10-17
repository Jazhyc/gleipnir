import json

from gleipnir.serving.bf16_monitor import CLASSIFIER_KEYS, with_model_text_config

CLASSIFIER = {
    "classifier_from_token": ["0", "1"],
    "is_causal": True,
    "method": "no_post_processing",
    "num_labels": 2,
}


def command(model, text_config):
    overrides = {"num_labels": 2, "text_config": {**text_config, **CLASSIFIER}}
    return ["serve", "--model", str(model), "--hf-overrides", json.dumps(overrides)]


def write_model(path, text_config):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"text_config": text_config}))


def test_matching_model_keeps_the_selected_command(tmp_path):
    text = {"num_hidden_layers": 32, "hidden_size": 2560}
    write_model(tmp_path / "4b", text)
    original = command(tmp_path / "4b", text)
    assert with_model_text_config(original) == original


def test_other_model_gets_its_own_text_config_and_the_classifier(tmp_path):
    write_model(tmp_path / "27b", {"num_hidden_layers": 64, "hidden_size": 5120})
    original = command(tmp_path / "27b", {"num_hidden_layers": 32, "hidden_size": 2560})
    result = with_model_text_config(original)
    overrides = json.loads(result[result.index("--hf-overrides") + 1])
    assert overrides["num_labels"] == 2
    assert overrides["text_config"] == {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        **{key: CLASSIFIER[key] for key in CLASSIFIER_KEYS},
    }
