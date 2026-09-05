"""Dataset extraction remains optional and diagnoses its missing dependency."""

import builtins

import pytest

from scripts import fault_codes_extract, text_to_sql_extract


@pytest.mark.parametrize("module, options", [
    (fault_codes_extract, {"source_limit": 1}),
    (text_to_sql_extract, {"limit": 1}),
])
@pytest.mark.parametrize("missing", ["datasets", "pyarrow"])
def test_data_loader_explains_missing_extra_without_masking_broken_install(
    monkeypatch, module, options, missing,
):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError(f"No module named {missing}", name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    expected = RuntimeError if missing == "datasets" else ModuleNotFoundError
    message = "uv run --locked --extra data" if missing == "datasets" else "pyarrow"
    with pytest.raises(expected, match=message):
        module.fetch_rows_from_datasets("unused", "unused", "train", **options)
