from experiments.serving.prefix_caching_tokens.compute import action_ends, work


def test_ood_rendering_checks_only_tool_calling_turns():
    t = "\nUSER: do it\n\nASSISTANT: Tool calls:\n- bash\n\nTOOL: ok\n\nASSISTANT: done"
    ends = action_ends(t)
    assert len(ends) == 1 and t[: ends[0]].endswith("- bash")


def test_bracket_rendering_ends_before_tool_output():
    t = (
        "\n[USER]\nx\n[ASSISTANT]\nhi\n[TOOL CALLS]\nls"
        "\n[TOOL]\nout\n[ASSISTANT]\nbye\n"
    )
    ends = action_ends(t)
    assert len(ends) == 1 and t[: ends[0]].endswith("[TOOL CALLS]\nls")


def test_xml_rendering_uses_assistant_blocks_with_functions():
    t = "<assistant><function>a</function></assistant><out/><assistant>no</assistant>"
    assert action_ends(t) == [t.index("<out/>")]


def test_cached_work_counts_each_prefix_once():
    prompt = (
        "I<agent_trajectory>\n\nASSISTANT: Tool calls:\n- a\n\nTOOL: x"
        "\n\nASSISTANT: Tool calls:\n- b\n\nTOOL: y</agent_trajectory>S"
    )
    counted = work(prompt)
    suffix = len("</agent_trajectory>S")
    first = prompt.index("- a") + 3
    second = prompt.index("- b") + 3
    assert counted["checks"] == 2
    assert counted["uncached"] == first + second + 2 * suffix
    assert counted["cached"] == second + 2 * suffix
