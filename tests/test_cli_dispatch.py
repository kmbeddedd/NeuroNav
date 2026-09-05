import sys

import pytest

import main as dispatcher
from scripts.benchmark import benchmark_ps08


def test_benchmark_options_reach_implementation(monkeypatch):
    received = []
    monkeypatch.setattr(benchmark_ps08, 'main', lambda argv: received.extend(argv))
    monkeypatch.setattr(sys, 'argv', ['main.py', 'benchmark', '--max-epochs', '2', '--device', 'cpu'])
    dispatcher.main()
    assert received[-4:] == ['--max-epochs', '2', '--device', 'cpu']


def test_unknown_benchmark_option_fails_before_training(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['main.py', 'benchmark', '--max-epoch', '2', '--typo'])
    with pytest.raises(SystemExit) as exc:
        dispatcher.main()
    assert exc.value.code == 2
