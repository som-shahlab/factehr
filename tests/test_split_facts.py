from factehr.utils import split_facts


def test_numbered_lists_with_sublists():
    text = """
    ## Independent Facts:
    1. Fact one.
    2. Fact two.
       - Subfact A.
       - Subfact B.
    3. Fact three.
    """
    expected_output = [
        "Fact one.",
        "Fact two.\n- Subfact A.\n- Subfact B.",
        "Fact three.",
    ]
    assert split_facts(text) == expected_output


def test_lists_with_header_non_numbered_item():
    text = """
    ## Independent Facts:
    * Fact one.
    * Fact two.
    * Fact three.
    """
    expected_output = ["Fact one.", "Fact two.", "Fact three."]
    assert split_facts(text) == expected_output


def test_lists_without_number_prefixes():
    text = """
    Fact one.
    Fact two.
    Fact three.
    """
    expected_output = ["Fact one.", "Fact two.", "Fact three."]
    assert split_facts(text) == expected_output
