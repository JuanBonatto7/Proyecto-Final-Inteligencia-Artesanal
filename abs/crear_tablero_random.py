from generateBoardImage import Board, BoardImageGenerator
from RandomOM import generar_tablero

# --- 1️⃣ Generar un tablero random ---
n = 5 # tamaño del tablero (6x6 por ejemplo)
tablero_random = generar_tablero(n)

# --- 2️⃣ Convertir al formato esperado por BoardImageGenerator ---
# Board espera una lista de listas de Tile (exactamente como devuelve generar_tablero)
board = Board(tiles=tablero_random)

# --- 3️⃣ Crear el generador de imágenes ---
generator = BoardImageGenerator(
    tiles_folder="./abs/tiles",  # ruta a las imágenes de tus losetas A.png, B.png, etc.
    tile_size=150            # tamaño de cada loseta (puedes ajustarlo)
)

# --- 4️⃣ Generar y guardar la imagen ---
imagen = generator.generate_board_image(board, "tablero_random.jpg")

# --- 5️⃣ Mostrar información ---
print("✅ Tablero aleatorio generado y guardado como 'tablero_random.jpg'")
