import py_compile
import unittest
from pathlib import Path


class AppSmokeTest(unittest.TestCase):
    def test_app_script_compiles(self):
        script_path = Path("app") / "main.py"
        py_compile.compile(script_path, doraise=True)


if __name__ == "__main__":
    unittest.main()
