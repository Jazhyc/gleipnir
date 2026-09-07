import pytest

from gleipnir.evaluation_lanes import adapter_lanes


def test_disjoint_balanced_adapter_assignment():
    assert adapter_lanes(["5", "10", "20", "50"]) == [["5", "20"], ["10", "50"]]
    assert adapter_lanes(["5"]) == [["5"]]


@pytest.mark.parametrize("names,n", [([], 2), (["a", "a"], 2), (["a"], 0)])
def test_invalid_assignment(names, n):
    with pytest.raises(ValueError):
        adapter_lanes(names, n)
