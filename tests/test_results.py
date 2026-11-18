import numpy as np
import pytest

from gmethods import RiskTable, __version__


class TestRiskTable:
    table = RiskTable(risks={"natural_course": 0.04, "never_x": 0.03}, label="t")

    def test_frame(self):
        f = self.table.to_frame()
        assert f.loc["never_x", "rd"] == pytest.approx(-0.01)
        assert f.loc["never_x", "rr"] == pytest.approx(0.75)
        assert f.loc["never_x", "avoided_per_1000"] == pytest.approx(10.0)

    def test_compare(self):
        truth = RiskTable(risks={"natural_course": 0.05, "never_x": 0.045})
        out = self.table.compare(truth)
        assert out.loc["never_x", "error"] == pytest.approx(-0.005)

    def test_missing_reference(self):
        with pytest.raises(KeyError):
            RiskTable(risks={"a": 0.1}).to_frame()

    def test_ci_columns(self):
        t = RiskTable(
            risks={"natural_course": 0.1, "b": 0.08}, ci={"b": (-0.03, -0.01)}
        )
        f = t.to_frame()
        assert np.isclose(f.loc["b", "rd_low"], -0.03)

    def test_repr(self):
        assert "never_x" in repr(self.table)


def test_version():
    assert __version__ == "0.1.0"
