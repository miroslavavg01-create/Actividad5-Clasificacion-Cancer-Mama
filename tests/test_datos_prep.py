import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fuentes"))
from datos_prep import obtener_datos_publicos, preparar_dataframe

class TestPreparacionDatos(unittest.TestCase):
    def test_dataset_publico_dimensiones(self):
        df = obtener_datos_publicos()
        self.assertEqual(len(df), 569)
        self.assertEqual(len(df.columns) - 1, 30)

    def test_limpieza_y_target(self):
        original = obtener_datos_publicos()
        limpio = preparar_dataframe(original)
        self.assertEqual(len(limpio), 569)
        self.assertFalse(limpio["target_maligno"].isna().any())
        self.assertSetEqual(set(limpio["target_maligno"].unique()), {0, 1})
        self.assertNotIn(" ", "".join(limpio.columns))

if __name__ == "__main__":
    unittest.main()
