import pytest
import pairstat_impl_rs


def test_sum_as_string():
    assert pairstat_impl_rs.sum_as_string(1, 1) == "2"
