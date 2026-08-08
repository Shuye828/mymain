from src.data.splits import assign_subjects, largest_remainder_counts


def test_largest_remainder_uses_every_subject() -> None:
    counts = largest_remainder_counts(
        105, {"train": 0.70, "validation": 0.15, "test": 0.15}
    )

    assert counts == {"train": 73, "validation": 16, "test": 16}
    assert sum(counts.values()) == 105


def test_subject_assignments_are_deterministic_and_disjoint() -> None:
    subjects = [str(index) for index in range(20)]
    kwargs = {
        "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "seed": 42,
        "dataset": "cpsc2021",
        "protocol": "source",
    }

    first = assign_subjects(subjects, **kwargs)
    second = assign_subjects(reversed(subjects), **kwargs)

    assert first == second
    assert set(first) == set(subjects)
    split_sets = [
        {subject for subject, split in first.items() if split == name}
        for name in ("train", "validation", "test")
    ]
    assert not (split_sets[0] & split_sets[1])
    assert not (split_sets[0] & split_sets[2])
    assert not (split_sets[1] & split_sets[2])
