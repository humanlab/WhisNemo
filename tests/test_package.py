"""Smoke tests for whisnemo package structure."""

import importlib


def test_version():
    import whisnemo
    assert whisnemo.__version__ == "0.1.0"


def test_core_helpers_importable():
    mod = importlib.import_module("whisnemo.core.helpers")
    assert hasattr(mod, "create_config")
    assert hasattr(mod, "get_words_speaker_mapping")
    assert hasattr(mod, "write_srt")


def test_postprocessing_importable():
    mod = importlib.import_module("whisnemo.postprocessing.remove_stutters")
    assert hasattr(mod, "correct_file_with_similarity")
    assert hasattr(mod, "process_folder")


def test_cli_importable():
    mod = importlib.import_module("whisnemo.cli.main")
    assert hasattr(mod, "main")
