"""Tests for retina.reddit."""



def test_reddit_module_imports() -> None:
    """Reddit scraper module can be imported."""
    import retina.reddit

    assert retina.reddit is not None
