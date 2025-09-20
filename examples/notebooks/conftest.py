#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2019-2025, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


from pathlib import Path

import jupytext
import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def pytest_collect_file(parent, file_path: Path):
    if file_path.suffix in (".ipynb", ".md") and "-Copy" not in file_path.name:
        return NotebookFile.from_parent(parent=parent, path=file_path)


class NotebookFile(pytest.File):
    @classmethod
    def from_parent(cls, parent, path):
        return super().from_parent(parent=parent, path=path)

    def collect(self):
        yield NotebookItem.from_parent(parent=self, name=self.path.name)


class NotebookItem(pytest.Item):
    def __init__(self, name, parent):
        super().__init__(name, parent)

    def runtest(self):
        # cur_dir = os.path.dirname(self.parent.path)

        # Load notebook (MyST .md or .ipynb)
        if self.parent.path.suffix == ".md":
            nb = jupytext.read(self.parent.path)
        else:
            with self.parent.path.open() as f:
                nb = nbformat.read(f, as_version=4)

        # ensure NotebookNode
        if not isinstance(nb, nbformat.NotebookNode):
            nb = nbformat.from_dict(nb)

        # ensure kernel info
        nb.metadata.setdefault(
            "kernelspec",
            {"name": "python3", "display_name": "Python 3", "language": "python"},
        )

        # clear previous outputs, if any
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None

        # execute notebook
        client = NotebookClient(
            nb,
            kernel_name="python3",
            timeout=600,
            iopub_timeout=600,
            allow_errors=False,
        )
        try:
            client.execute()
        except CellExecutionError as e:
            raise NotebookException(e) from e

        # write outputs to .ipynb in any case
        if self.parent.path.suffix == ".md":
            # as .md.iypnb
            # out_path = self.parent.path.with_suffix(self.parent.path.suffix + ".ipynb")
            # as .iypnb
            out_path = self.parent.path.with_suffix(".ipynb")
        else:
            out_path = self.parent.path

        # output notebooks in render
        out_path_parts = tuple(
            "render" if part == "notebooks" else part for part in out_path.parts
        )
        out_path = Path(*out_path_parts)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, NotebookException):
            return excinfo.exconly()
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.parent.path, 0, f"TestCase: {self.name}"


class NotebookException(Exception):
    pass
