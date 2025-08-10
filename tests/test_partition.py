# this is designed for testing the partition algorithm

from pairstat._partition_cy import build_task_it_factory
from itertools import product


class DummyAccumulator:
    def __init__(self):
        self.pairs = set()

    def add_pair_entry(self, a, b):
        if a <= b:
            val = (a, b)
        else:
            val = (b, a)
        # assert val not in self.pairs
        self.pairs.add(val)

    def assert_equal(self, other):
        assert len(self.pairs.symmetric_difference(other.pairs)) == 0


def _apply_accumulator(accum, a_vals, b_vals=[], task_iter=None):
    def apply_auto(container):
        length = len(container)
        for i in range(length):
            for j in range(i + 1, length):
                accum.add_pair_entry(container[i], container[j])

    def apply_cross(iter_a, iter_b):
        for a, b in product(iter_a, iter_b):
            accum.add_pair_entry(a, b)

    if task_iter is None:
        if len(b_vals) == 0:
            apply_auto(a_vals)
        else:
            apply_cross(a_vals, b_vals)
    else:
        for elem in task_iter:
            if len(b_vals) == 0:
                if elem.start_B == elem.stop_B == 0:
                    apply_auto(a_vals[elem.start_A : elem.stop_A])
                else:
                    apply_cross(
                        a_vals[elem.start_A : elem.stop_A],
                        a_vals[elem.start_B : elem.stop_B],
                    )
            else:
                apply_cross(
                    a_vals[elem.start_A : elem.stop_A],
                    b_vals[elem.start_B : elem.stop_B],
                )


def _get_ref(a_vals, b_vals):
    ref_vals = DummyAccumulator()
    _apply_accumulator(ref_vals, a_vals=a_vals, b_vals=b_vals, task_iter=None)
    return ref_vals


def _run_test(a_vals, b_vals, nproc, ref_vals=None, assert_within_max_nproc=False):
    if ref_vals is None:
        ref_vals = _get_ref(a_vals, b_vals)

    factory = build_task_it_factory(
        nproc=nproc,
        n_points=len(a_vals),
        n_points_other=len(b_vals),
        skip_small_prob_check=True,
    )
    actual_vals = DummyAccumulator()

    max_nproc = factory.num_partitions()
    if max_nproc < nproc:
        if assert_within_max_nproc:
            raise AssertionError("nproc exceeds number of partitions")
        else:
            print(
                f"Number of partitions, {max_nproc} exceeds nproc, {nproc}. "
                "Adjusting nproc accordingly"
            )
            nproc = max_nproc

    for proc_id in range(nproc):
        tmp = factory.build_iterator(proc_id)
        _apply_accumulator(actual_vals, a_vals, b_vals, task_iter=tmp)
    ref_vals.assert_equal(actual_vals)


def test_cross_partition():
    locase = "abcdefghijklmnopqrstuvwxyz"
    upcase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    MAX_LEN = 26

    for shorter_len in range(1, MAX_LEN + 1):
        for longer_len in range(max(shorter_len, 4), MAX_LEN + 1):
            print(shorter_len, longer_len)
            shorter, longer = locase[:shorter_len], upcase[:longer_len]
            ref_vals = _get_ref(shorter, longer)
            # print(ref_vals.pairs)
            alt_ref_vals = _get_ref(longer, shorter)
            ref_vals.assert_equal(alt_ref_vals)
            for nproc in range(1, longer_len):
                print(nproc)
                _run_test(
                    shorter,
                    longer,
                    nproc,
                    ref_vals=ref_vals,
                    assert_within_max_nproc=True,
                )


def test_auto_partition():
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for length in range(2, len(letters) + 1):
        print(f"length = {length}")
        vals = letters[:length]
        ref_vals = _get_ref(vals, [])

        for nproc in range(1, 61):  # hard limit in the code for now is 60
            _run_test(vals, [], nproc, ref_vals=ref_vals)
