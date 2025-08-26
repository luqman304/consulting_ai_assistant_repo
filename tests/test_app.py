import os

def test_repo_contains_files():
    expected = ["requirements.txt", "sample_case_study.pdf"]
    for f in expected:
        assert os.path.exists(f), f"{f} missing"
