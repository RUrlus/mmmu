import numpy as np
import mmu


def test_add():
    assert mmu.core.add(1, 1) == 2


def test_arange():
    ubs = np.random.randint(2, 1000, size=100)
    for ub in ubs:
        step = np.random.randint(1, ub)
        np.testing.assert_equal(
            mmu.core.arange(0, ub, step=step), np.arange(ub, step=step)
        )
