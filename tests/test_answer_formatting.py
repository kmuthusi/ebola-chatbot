from answer_formatting import format_answer_markdown


def test_inline_bullets_render_as_separate_markdown_items():
    answer = "Care includes: • Give fluids [guide.pdf, page 2]. • Monitor symptoms [guide.pdf, page 3]."
    assert format_answer_markdown(answer) == (
        "Care includes:\n\n- Give fluids [guide.pdf, page 2].\n\n"
        "- Monitor symptoms [guide.pdf, page 3]."
    )


def test_inline_numbering_render_as_separate_markdown_items():
    answer = "Steps: 1. First [guide.pdf, page 2]. 2. Second [guide.pdf, page 3]."
    assert format_answer_markdown(answer) == (
        "Steps:\n\n1. First [guide.pdf, page 2].\n\n"
        "2. Second [guide.pdf, page 3]."
    )


def test_citation_on_next_line_stays_with_previous_sentence():
    answer = "A blood test is used.\n\n[guide.pdf, page 2]"
    assert format_answer_markdown(answer) == "A blood test is used. [guide.pdf, page 2]"
