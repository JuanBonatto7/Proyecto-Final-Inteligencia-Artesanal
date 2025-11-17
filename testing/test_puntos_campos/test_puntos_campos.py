import unittest

from modules.CarcassoneFieldsv5.puntos_campos import calculate_field_scores


class TestMathOperations(unittest.TestCase):

    def test_tablero1(self):
        resultado = calculate_field_scores("testing/casos_tableros_generados/tablero1.jpg")
        resultado_esperado = {1:0,2:21}

        self.assertEqual(resultado,resultado_esperado)

    def test_tablero2(self):
        resultado = calculate_field_scores("testing/casos_tableros_generados/tablero2.jpg")
        resultado_esperado = {1:0,2:9}

        self.assertEqual(resultado,resultado_esperado)
    
    def test_tablero3(self):
        resultado = calculate_field_scores("testing/casos_tableros_generados/tablero3.jpg")
        resultado_esperado = {1:6,2:0}

        self.assertEqual(resultado,resultado_esperado)

    def test_tablero4(self):
        resultado = calculate_field_scores("testing/casos_tableros_generados/tablero4.jpg")
        resultado_esperado = {1:0,2:12}

        self.assertEqual(resultado,resultado_esperado)


if __name__ == '__main__':
    unittest.main()