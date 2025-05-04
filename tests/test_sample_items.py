import pytest

from selectron.util.sample_items import sample_items


@pytest.fixture
def sample_data():
    return [
        {"a": 1, "b": 2},  # Shape 1
        {"a": 3, "b": 4},  # Shape 1
        {"a": 5, "c": 6},  # Shape 2
        {"a": 7, "c": 8},  # Shape 2
        {"a": 9, "c": 10},  # Shape 2
        {"d": 11},  # Shape 3
        {"a": 12, "b": 13, "c": 14},  # Shape 4
        {"a": 15, "b": 16},  # Shape 1
    ]


def test_basic_sampling(sample_data):
    """Test basic sampling with default size."""
    samples = sample_items(sample_data)  # default size 4
    assert len(samples) == 4
    # Check that distinct shapes are prioritized
    # Extract dicts from tuples for shape checking
    sampled_dicts = [s[1] for s in samples]
    shapes = {frozenset(item.keys()) for item in sampled_dicts}
    assert len(shapes) == 4  # Should get one of each shape


def test_sampling_small_size(sample_data):
    """Test sampling with size smaller than distinct shapes."""
    samples = sample_items(sample_data, sample_size=2)
    assert len(samples) == 2
    sampled_dicts = [s[1] for s in samples]
    shapes = {frozenset(item.keys()) for item in sampled_dicts}
    assert len(shapes) == 2  # Should have 2 distinct shapes


def test_sampling_large_size(sample_data):
    """Test sampling with size larger than items."""
    samples = sample_items(sample_data, sample_size=10)
    assert len(samples) == len(sample_data)  # Should return all items
    # Ensure no duplicates unless input had duplicates
    # Check ids of the returned dicts
    sampled_dicts = [s[1] for s in samples]
    assert len({id(item) for item in sampled_dicts}) == len(sampled_dicts)


def test_sampling_exact_size_distinct(sample_data):
    """Test sampling size equal to distinct shapes."""
    samples = sample_items(sample_data, sample_size=4)
    assert len(samples) == 4
    sampled_dicts = [s[1] for s in samples]
    shapes = {frozenset(item.keys()) for item in sampled_dicts}
    assert len(shapes) == 4


def test_sampling_size_between_distinct_and_total(sample_data):
    """Test sampling size between distinct shapes and total items."""
    samples = sample_items(sample_data, sample_size=6)
    assert len(samples) == 6
    sampled_dicts = [s[1] for s in samples]
    shapes = {frozenset(item.keys()) for item in sampled_dicts}
    assert len(shapes) == 4  # Should still represent all shapes
    # Ensure no duplicates unless input had duplicates
    assert len({id(item) for item in sampled_dicts}) == len(sampled_dicts)


def test_empty_input():
    """Test with empty input list."""
    assert sample_items([]) == []


def test_zero_sample_size(sample_data):
    """Test with sample_size=0."""
    assert sample_items(sample_data, sample_size=0) == []


def test_non_serializable_items():
    """Test with items that cannot be JSON serialized (should be skipped)."""
    data = [
        {"a": 1},
        {"b": {1, 2}},  # Non-serializable
        {"c": 3},
        {"d": complex(1, 2)},  # Non-serializable
        {"e": 5},
    ]
    samples = sample_items(data, sample_size=4)
    assert len(samples) == 3  # Only serializable items are sampled
    for item_tuple in samples:
        # Check the value in the dictionary part of the tuple
        assert isinstance(list(item_tuple[1].values())[0], int)


def test_all_non_serializable():
    """Test when all items are non-serializable."""
    data = [{"b": {1, 2}}, {"d": complex(1, 2)}]
    assert sample_items(data) == []


def test_single_item():
    """Test with only one item."""
    data = [{"a": 1}]
    samples = sample_items(data, sample_size=1)
    assert len(samples) == 1
    assert samples[0][0] == 0  # Check index
    assert samples[0][1] == {"a": 1}  # Check dict

    samples_large = sample_items(data, sample_size=5)
    assert len(samples_large) == 1
    assert samples_large[0][0] == 0
    assert samples_large[0][1] == {"a": 1}


def test_identical_items():
    """Test with list of identical items."""
    data = [{"a": 1}, {"a": 1}, {"a": 1}]
    samples = sample_items(data, sample_size=2)
    assert len(samples) == 2
    # Check the dictionary parts
    assert samples[0][1] == {"a": 1}
    assert samples[1][1] == {"a": 1}
    # Check that indices are distinct
    assert samples[0][0] != samples[1][0]
    assert samples[0][0] in [0, 1, 2]
    assert samples[1][0] in [0, 1, 2]

    sampled_dicts = [s[1] for s in samples]
    shapes = {frozenset(item.keys()) for item in sampled_dicts}
    assert len(shapes) == 1
