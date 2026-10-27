"""Small CPU checks for the standalone release interfaces."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleasePathTests(unittest.TestCase):
    def test_no_private_humaneval_imports(self):
        for path in ROOT.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("benchmark.humaneval."), path)

    def test_daedal_generation_keywords_match(self):
        sampler = ast.parse((ROOT / "daedal_cal/daedal_cal.py").read_text())
        generate = next(n for n in sampler.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
        parameters = {arg.arg for arg in generate.args.args}
        runner = ast.parse((ROOT / "daedal_cal/daedal_code_infilling.py").read_text())
        calls = [n for n in ast.walk(runner) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "generate"]
        self.assertEqual(len(calls), 1)
        self.assertLessEqual({kw.arg for kw in calls[0].keywords}, parameters)

    def test_stratified_reader_uses_explicit_result_path(self):
        path = ROOT / "benchmark/humaneval_infilling/analysis_by_oracle_length.py"
        spec = importlib.util.spec_from_file_location("stratified_analysis", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            record = {"task_id": "test", "passed": True}
            (run / "samples_results.jsonl").write_text(json.dumps(record) + "\n")
            self.assertEqual(module.load_results(directory, "run/samples_results.jsonl"), [record])


if __name__ == "__main__":
    unittest.main()
