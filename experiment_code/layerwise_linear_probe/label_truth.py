"""Exact answers for the six probe templates, and the "fixed" correctness rule.

correction rerun: several generators (question_generate.py) build the label from rounded
or floored intermediate values, so the stated problem often has a non-integer exact
answer. 922 floors each male count, 584 floors one-third of the king-cab price, 884 and
942 round the final value. A model that rounds the exact value and a model that floors
each step can both be right. The fixed rule accepts either:

  correct  <=>  answer is an integer within 1 of the exact value, or
                answer == generator label and the label is a valid reading

The label is a valid reading where the generator floors a non-integer intermediate count
(922: a male count; 584: one-third of the king-cab price), since whole students and
whole dollars are a fair interpretation. Elsewhere an integer exact answer must match
exactly, even against the label: the 884 labels truncate a float (6.5 dozen * 12 =
77.99... -> 77), so 28 rows with an integer exact answer carry a label 1 too low.

Run `python3 label_truth.py` to check every q_data row against its generator label.
"""
import math
import re
from fractions import Fraction as F

_UNIT = {"half": 2, "third": 3, "fourth": 4, "fifth": 5}
_WORDS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def _n(s):
    return int(s.replace(",", ""))


def _frac(s):
    m = re.fullmatch(r"(\d+)/(\d+)", s)
    return F(int(m[1]), int(m[2])) if m else F(1, _UNIT[s.split("-")[1]])


def exact_value(q):
    """The exact answer to the question text, as a Fraction."""
    if q.startswith("Brook Hills"):  # 922
        m = re.search(r"enrolls ([\d,]+) students.*?and (\S+) of the students over.*?and (\S+) of the students under", q)
        total, half = _n(m[1]), _n(m[1]) // 2
        return total - half * _frac(m[2]) - half * _frac(m[3])
    if q.startswith("Bill is ordering"):  # 584
        base, king, _, light = [_n(x) for x in re.findall(r"\$([\d,]+)", q)[:4]]
        leather = F(king, 3)
        return base + king + leather + (leather - 500) + light
    if "dozen" in q:  # 884
        init = _WORDS.index(re.search(r"herself; (\w+) dozen", q)[1])
        office = F(*map(int, re.search(r"home with (\d+)/(\d+) dozen", q).groups()))
        surprise = int(re.search(r"had (\d+) dozen brownies waiting", q)[1])
        w, a, b = map(int, re.search(r"(\d+) (\d+)/(\d+) dozen brownies were eaten", q).groups())
        return (init + office + surprise - (w + F(a, b))) * 12
    if "homework" in q:  # 942
        p = [int(x) for x in re.findall(r"(\d+)% chance", q)[:3]]
        return F(100 - p[0], 100) * F(100 - p[1], 100) * F(100 - p[2], 100) * 100
    if "newspaper" in q:  # 270
        return F(int(re.search(r"perhaps page (\d+) is printed", q)[1]), 4)
    if "river" in q:  # 785
        init, inc = [_n(x) for x in re.findall(r"([\d,]{4,}) gallons", q)[:2]]
        return F(init * 2 + inc)
    raise ValueError("unknown template: " + q[:60])


def label_is_reading(q):
    """True where the label floors a non-integer intermediate count (922, 584)."""
    if q.startswith("Brook Hills"):
        m = re.search(r"enrolls ([\d,]+) students.*?and (\S+) of the students over.*?and (\S+) of the students under", q)
        half = _n(m[1]) // 2
        return (half * _frac(m[2])).denominator != 1 or (half * _frac(m[3])).denominator != 1
    if q.startswith("Bill is ordering"):
        return _n(re.findall(r"\$([\d,]+)", q)[1]) % 3 != 0
    return False


def is_correct_fixed(answer, label, exact, reading=False):
    """The fixed rule. answer and label are numbers, exact a Fraction, reading from label_is_reading."""
    if answer is None or (isinstance(answer, float) and math.isnan(answer)):
        return False
    if float(answer).is_integer() and abs(F(int(answer)) - exact) < 1:
        return True
    return answer == label and reading


def _check():
    import csv
    import glob
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for f in sorted([os.path.join(here, "q_data", f"{t}_1000.csv") for t in (270, 584, 785, 884, 922, 942)]):
        rows = list(csv.DictReader(open(f)))
        nonint = wrong = 0
        for r in rows:
            e, g = exact_value(r["question"]), int(r["answer"])
            nonint += e.denominator != 1
            assert abs(F(g) - e) < 2, (f, r["index"], g, e)  # generator rounds or floors, never more
            wrong += not is_correct_fixed(g, g, e, label_is_reading(r["question"]))  # label fails the rule
        print(os.path.basename(f), "rows", len(rows), "non-integer exact answers", nonint, "wrong labels", wrong)
        assert wrong == (28 if f.endswith("884_1000.csv") else 0)
    e = F(30625, 10)  # 3062.5
    assert is_correct_fixed(3062, 3064, e) and is_correct_fixed(3063, 3064, e) and is_correct_fixed(3064, 3064, e, True)
    assert not is_correct_fixed(3064, 3064, e)  # label 1.5 off is accepted only as a valid reading
    assert not is_correct_fixed(3061, 3064, e) and not is_correct_fixed(3062.5, 3064, e)
    assert is_correct_fixed(7, 7, F(7)) and not is_correct_fixed(8, 7, F(7))
    assert is_correct_fixed(78, 77, F(78)) and not is_correct_fixed(77, 77, F(78))  # 884 bad label
    print("label_truth self-check ok")


if __name__ == "__main__":
    _check()
