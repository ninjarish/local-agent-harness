import tempfile
import unittest
import zipfile
from pathlib import Path

from local_agent.tools import ToolError, WorkspaceFiles, calculate, read_document


class CalculatorTests(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(calculate("(287 * 43) + 19"), 12360)

    def test_rejects_function_calls(self):
        with self.assertRaises(ToolError):
            calculate("__import__('os').system('whoami')")


class WorkspaceFilesTests(unittest.TestCase):
    def test_reads_and_writes_inside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            files = WorkspaceFiles(Path(directory))
            files.write_text("notes/test.txt", "hello")
            self.assertEqual(files.read_text("notes/test.txt"), "hello")
            self.assertEqual(files.list_files("notes"), ["notes\\test.txt"])

    def test_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            files = WorkspaceFiles(Path(directory))
            with self.assertRaises(ToolError):
                files.read_text("../outside.txt")


class DocumentTests(unittest.TestCase):
    def test_extracts_docx_paragraph_text(self):
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Migration overview</w:t></w:r></w:p>
    <w:p><w:r><w:t>Validate the target environment.</w:t></w:r></w:p>
  </w:body>
</w:document>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            self.assertEqual(
                read_document(str(path)),
                "Migration overview\nValidate the target environment.",
            )

    def test_rejects_non_docx_file(self):
        with self.assertRaises(ToolError):
            read_document("notes.txt")


if __name__ == "__main__":
    unittest.main()
