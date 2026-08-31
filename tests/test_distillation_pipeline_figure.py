from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.plot_distillation_pipeline_comparison import (
    GLEIPNIR_STAGES,
    PAPER_STAGES,
    plot_pipeline_comparison,
)


def _figure_text(figure: object) -> str:
    texts = [text.get_text() for text in figure.texts]
    for axis in figure.axes:
        texts.extend(text.get_text() for text in axis.texts)
    return "\n".join(texts)


def test_pipeline_contract_keeps_ground_truth_in_corrective_retry() -> None:
    assert "4 structured rationales" in PAPER_STAGES[0]
    assert "using ground-truth label" in PAPER_STAGES[2]
    assert "Claude Sonnet 4.5" in PAPER_STAGES[3]
    assert "p(0),  p(1)" in GLEIPNIR_STAGES[0]
    assert len(PAPER_STAGES) > len(GLEIPNIR_STAGES)


def test_pipeline_figure_contains_intended_comparison_without_epoch_details() -> None:
    figure = plot_pipeline_comparison()
    text = _figure_text(figure)

    assert "Filtered rationale distillation" in text
    assert "Direct decision-distribution distillation" in text
    assert "using ground-truth label" in text
    assert "0 = benign  ·  1 = misaligned" in text
    assert "private reasoning is excluded" in text
    assert "Deployable action-only monitor" not in text
    assert "epoch" not in text.lower()
    assert "learning rate" not in text.lower()


def test_pipeline_figure_uses_vector_boxes_and_arrows() -> None:
    figure = plot_pipeline_comparison()
    patches = figure.axes[0].patches

    assert sum(isinstance(patch, FancyBboxPatch) for patch in patches) >= 12
    assert sum(isinstance(patch, FancyArrowPatch) for patch in patches) >= 9
