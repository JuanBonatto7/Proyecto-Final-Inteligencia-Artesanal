import unittest
import sys
import os
from pathlib import Path

# Asegura que el raíz del proyecto esté en sys.path al ejecutar este archivo directamente
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.CarcassoneFieldsv5.puntos_campos import calculate_field_scores


class TestMathOperations(unittest.TestCase):

    def test_tablero1(self):
        resultado = calculate_field_scores("../casos_tableros_generados/tablero1.jpg")
        resultado_esperado = {1:0,2:21}

        self.assertEqual(resultado,resultado_esperado)

    def test_tablero2(self):
        resultado = calculate_field_scores("../casos_tableros_generados/tablero2.jpg")
        resultado_esperado = {1:0,2:9}

        self.assertEqual(resultado,resultado_esperado)
    
    def test_tablero3(self):
        resultado = calculate_field_scores("../casos_tableros_generados/tablero3.jpg")
        resultado_esperado = {1:6,2:0}

        self.assertEqual(resultado,resultado_esperado)

    def test_tablero4(self):
        resultado = calculate_field_scores("../casos_tableros_generados/tablero4.jpg")
        resultado_esperado = {1:0,2:12}

        self.assertEqual(resultado,resultado_esperado)

    def test_tablero5(self):
        resultado = calculate_field_scores("../casos_tableros_generados/tablero5.jpg")
        resultado_esperado = {1:9,2:6}

        self.assertEqual(resultado,resultado_esperado)

    def test_tablero6(self):
        resultado = calculate_field_scores("../casos_tableros_generados/tablero6.jpg")
        resultado_esperado = {1:0,2:15}

        self.assertEqual(resultado,resultado_esperado)

    def test_tablero7(self):
        resultado = calculate_field_scores("../casos_tableros_generados/tablero7.jpg")
        resultado_esperado = {1:3,2:9}

        self.assertEqual(resultado,resultado_esperado)


if __name__ == '__main__':
    unittest.main()