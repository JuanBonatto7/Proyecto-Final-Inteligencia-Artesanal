from PIL import Image, ImageDraw #pillow
import os #para trabajar con rutas de archivos
from dataclasses import dataclass #decorador para las clases pa
from typing import Tuple, List, Optional #para algunas notaciones

@dataclass
class Tile:
    """Representa una tile pa"""
    nombre: str  # tipo de tile (A..X)
    orientacion: int  # 0, 1, 2, 3 (rotación en múltiplos de 90°)
    meeple: Tuple[int, int]  # (jugador, posición) 
                              # jugador: 0=ninguno, 1=jugador1, 2=jugador2
                              # posición: 0-8 (ver mapa de posiciones)



@dataclass
class Board:
    """Representa el tablero completo del tiles paaa"""
    tiles: List[List[Optional[Tile]]]  # matriz de losetas (puede tener None(Optional))






class BoardImageGenerator:
    """Generador de imágenes de tableros de Carcassonne"""
    
    # mapa de posiciones de meeples (coordenadas relativas 0.0-1.0)
    MEEPLE_POSITIONS = {
        0: (0.20, 0.20),  # Noroeste
        1: (0.50, 0.15),  # Norte
        2: (0.80, 0.20),  # Noreste
        3: (0.15, 0.50),  # Oeste
        4: (0.50, 0.50),  # Centro
        5: (0.85, 0.50),  # Este
        6: (0.20, 0.80),  # Suroeste
        7: (0.50, 0.85),  # Sur
        8: (0.80, 0.80),  # Sureste
    }


    
    def __init__(self, tiles_folder: str, tile_size: int = 200):
        """
        inicializa el generador de imagenes de tablerou
        
        Args:
            tiles_folder: Ruta a la carpeta con las imágenes PNG/JPG de los tiles (A.png hasta X.png)
            tile_size: Tamaño en píxeles para normalizar cada loseta (default: 200)
        """
        self.tiles_folder = tiles_folder
        self.tile_size = tile_size
        self.meeple_colors = {
            1: (255, 0, 0),    # rojo para jugador 1
            2: (0, 0, 255),    # azul para jugador 2
        }
        self._tile_cache = {}  # cache para imágenes cargadas
        


    def _find_tile_file(self, tile_name: str) -> Optional[str]:
        """
        busca el archivo de imagen del tile (PNG o JPG)
        
        Args:
            tile_name: Nombre del tile (letra A-X, 24 tiles en total)
            
        Returns:
            ruta completa del archivo o None si no existe
        """
        for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            tile_path = os.path.join(self.tiles_folder, f"{tile_name}{extension}")
            if os.path.exists(tile_path):
                return tile_path
        return None
    


    def _create_placeholder_tile(self, tile_name: str) -> Image.Image:
        """
        Crea una loseta placeholder cuando no se encuentra la imagen
        
        Args:
            tile_name: Nombre del tile para mostrar
            
        Returns:
            Imagen placeholder
        """
        img = Image.new('RGB', (self.tile_size, self.tile_size), color='lightgray')
        #esto crea una imagen desde cero (modo de color,(tamañoX,tamañoY),color de fondo)

        draw = ImageDraw.Draw(img)
        #instancia como un "lapiz" para dibujar sobre img
        
        # pinto un borde
        draw.rectangle([0, 0, self.tile_size-1, self.tile_size-1], outline='black', width=3)
        
        # Texto centrado
        text_size = self.tile_size // 3
        bbox = draw.textbbox((0, 0), tile_name) #calculo el bounding box, devuelvo (x1,y1,x2,y2)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.tile_size - text_width) // 2
        y = (self.tile_size - text_height) // 2
        draw.text((x, y), tile_name, fill='black') #dibujo texto en la imagen
        
        return img
    


    def load_tile_image(self, tile_name: str) -> Image.Image:
        """
        Carga y normaliza la imagen de una loseta
        
        Args:
            tile_name: Nombre del tile (A-X, 24 tiles)
            
        Returns:
            Imagen normalizada al tamaño configurado
        """
        # Verificar cache
        if tile_name in self._tile_cache:
            return self._tile_cache[tile_name].copy()
        
        # Buscar archivo
        tile_path = self._find_tile_file(tile_name)
        
        if tile_path is None:
            print(f"⚠ Advertencia: No se encontró imagen para tile '{tile_name}', usando placeholder")
            img = self._create_placeholder_tile(tile_name) #si no encuentro la imagen hago imagen placeholder
        else:
            try:
                # Cargar imagen
                img = Image.open(tile_path)
                
                # Convertir a RGB si es necesario (por si tiene canal alpha)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize manteniendo proporción y centrando si es necesario
                img = self._resize_and_center(img)
                
            except Exception as e:
                print(f"⚠ Error al cargar '{tile_path}': {e}")
                img = self._create_placeholder_tile(tile_name)
        
        # Guardar en cache
        self._tile_cache[tile_name] = img.copy()
        return img
    


    def _resize_and_center(self, img: Image.Image) -> Image.Image:
        """
        Redimensiona la imagen al tamaño objetivo manteniendo proporción
        y centra si es necesario
        
        Args:
            img: Imagen original (cualquier tamaño)
            
        Returns:
            Imagen redimensionada y centrada
        """
        # Calcular proporción
        width, height = img.size
        aspect_ratio = width / height
        
        if aspect_ratio > 1:  # Más ancha que alta
            new_width = self.tile_size
            new_height = int(self.tile_size / aspect_ratio)
        else:  # Más alta que ancha o cuadrada
            new_height = self.tile_size
            new_width = int(self.tile_size * aspect_ratio)
        
        # Redimensionar
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Si no es cuadrada, centrar en canvas cuadrado
        if new_width != self.tile_size or new_height != self.tile_size:
            canvas = Image.new('RGB', (self.tile_size, self.tile_size), color='white')
            offset_x = (self.tile_size - new_width) // 2
            offset_y = (self.tile_size - new_height) // 2
            canvas.paste(img_resized, (offset_x, offset_y))
            return canvas
        
        return img_resized
    
    def rotate_tile(self, img: Image.Image, orientation: int) -> Image.Image:
        """
        Rota la imagen según la orientación
        
        Args:
            img: Imagen a rotar
            orientation: 0=0°, 1=90°, 2=180°, 3=270°
            
        Returns:
            Imagen rotada
        """
        angle = orientation * 90
        return img.rotate(-angle, expand=False)
    


    def get_meeple_position(self, position: int) -> Tuple[int, int]:
        """
        Calcula las coordenadas absolutas del meeple
        
        Args:
            position: Posición del meeple (0-8)
            
        Returns:
            Tupla (x, y) en píxeles
        """
        rel_x, rel_y = self.MEEPLE_POSITIONS.get(position, (0.5, 0.5))
        x = int(rel_x * self.tile_size)
        y = int(rel_y * self.tile_size)
        return (x, y)
    

    
    def draw_meeple(self, img: Image.Image, player: int, position: int) -> Image.Image:
        """
        Dibuja un meeple en la loseta
        
        Args:
            img: Imagen de la loseta
            player: Número de jugador (0=ninguno, 1, 2, etc.)
            position: Posición del meeple (0-8)
            
        Returns:
            Imagen con el meeple dibujado
        """
        if player == 0:
            return img
        
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        x, y = self.get_meeple_position(position)
        radius = self.tile_size // 10
        color = self.meeple_colors.get(player, (128, 128, 128))  # Gris por defecto
        
        # Cabeza del meeple
        head_radius = radius // 2
        draw.ellipse(
            [x - head_radius, y - radius, 
             x + head_radius, y - radius + head_radius * 2], 
            fill=color, outline='black', width=1
        )
        
        # Cuerpo del meeple
        body_width = radius
        body_height = radius
        draw.ellipse(
            [x - body_width, y - radius // 3, 
             x + body_width, y + body_height], 
            fill=color, outline='black', width=2
        )
        
        return img_copy
    
    def generate_board_image(self, board: Board, output_path: str = "tablero.jpg") -> Image.Image:
        """
        Genera la imagen completa del tablero
        
        Args:
            board: Objeto Board con la matriz de tiles
            output_path: Ruta donde guardar la imagen
            
        Returns:
            Imagen generada
        """
        rows = len(board.tiles)
        cols = len(board.tiles[0]) if rows > 0 else 0
        
        # Crear canvas del tablero
        board_width = cols * self.tile_size
        board_height = rows * self.tile_size
        board_img = Image.new('RGB', (board_width, board_height), color='white')
        
        # Procesar cada loseta
        for i, row in enumerate(board.tiles):
            for j, tile in enumerate(row):
                if tile is None:
                    continue
                
                # Cargar y procesar tile
                tile_img = self.load_tile_image(tile.nombre)
                tile_img = self.rotate_tile(tile_img, tile.orientacion)
                tile_img = self.draw_meeple(tile_img, tile.meeple[0], tile.meeple[1])
                
                # Pegar en el tablero
                x = j * self.tile_size
                y = i * self.tile_size
                board_img.paste(tile_img, (x, y))
        
        # Guardar
        board_img.save(output_path, 'JPEG', quality=95)
        
        # Información
        print(f"✓ Tablero generado exitosamente")
        print(f"  Archivo: {output_path}")
        print(f"  Dimensiones: {board_width}x{board_height} píxeles")
        print(f"  Tablero: {rows}x{cols} losetas")
        print(f"  Tiles en cache: {len(self._tile_cache)}")
        
        return board_img
    
    def add_player_color(self, player_number: int, color: Tuple[int, int, int]):
        """
        Añade un color personalizado para un jugador
        
        Args:
            player_number: Número del jugador (1, 2, 3, etc.)
            color: Tupla RGB (r, g, b)
        """
        self.meeple_colors[player_number] = color


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GENERADOR DE TABLERO DE CARCASSONNE")
    print("=" * 60)
    
    # Crear tablero de ejemplo
    ejemplo_tiles = [
        # FILA 0
        [
            None,              # Col 0
            None,              # Col 1
            None,              # Col 2
            None,              # Col 3
            Tile("W", 0, (0, 0)),  # Col 4
            Tile("D", 1, (0, 0)),  # Col 5
            None,                # Col 6
            None,              # Col 7
            None,              # Col 8
            None,              # Col 9
            None,              # Col 10
            None,              # Col 11
        ],

        # FILA 1
        [
            None,              # Col 0
            Tile("C", 0, (0, 0)),  # Col 1
            None,              # Col 2
            None,              # Col 3
            Tile("L", 0, (0, 0)),  # Col 4
            Tile("N", 0, (0, 0)),  # Col 5
            Tile("N", 2, (0, 0)),  # Col 6
            Tile("T", 3, (0, 0)),  # Col 7
            Tile("S", 1, (0, 0)),              # Col 8
            Tile("H", 0, (0, 0)),              # Col 9
            Tile("I", 2, (0, 0)),              # Col 10
            None,              # Col 11
        ],

        # FILA 2
        [
            None,              # Col 0
            Tile("S", 3, (0, 0)),  # Col 1
            Tile("V", 0, (0, 0)),  # Col 2
            None,              # Col 3
            Tile("W", 2, (0, 0)),  # Col 4
            Tile("V", 0, (0, 0)),  # Col 5
            None,               # Col 6
            Tile("P", 0, (0, 0)),  # Col 7
            None,           # Col 8
            Tile("E", 2, (0, 0)),              # Col 9
            None,              # Col 10
            None,              # Col 11
        ],

        # FILA 3
        [
            None,              # Col 0
            None,  # Col 1
            Tile("W", 3, (0, 0)),  # Col 2
            Tile("U", 1, (0, 0)),  # Col 3
            Tile("U", 1, (0, 0)),  # Col 4
            Tile("P", 0, (0, 0)),  # Col 5
            Tile("M", 3, (0, 0)),              # Col 6
            None,  # Col 7
            Tile("U", 0, (0, 0)),  # Col 8
            Tile("H", 1, (0, 0)),              # Col 9
            None,              # Col 10
            None,              # Col 11
        ],

        # FILA 4
        [
            None,  # Col 0
            None,  # Col 1
            Tile("U", 0, (0, 0)),  # Col 2
            Tile("V", 3, (0, 0)),  # Col 3
            Tile("V", 0, (0, 0)),  # Col 4
            Tile("B", 0, (0, 0)),  # Col 5
            Tile("E", 0, (0, 0)),  # Col 6
            Tile("V", 3, (0, 0)),              # Col 7
            Tile("K", 0, (0, 0)),  # Col 8
            Tile("R", 0, (0, 0)),              # Col 9
            Tile("P", 0, (0, 0)),              # Col 10
            Tile("J", 2, (0, 0)),              # Col 11
        ],

        # FILA 5
        [
            Tile("A", 0, (0, 0)),  # Col 0 - Con meeple negro
            None,  # Col 1
            Tile("U", 0, (0, 0)),  # Col 2
            Tile("V", 2, (0, 0)),              # Col 3
            Tile("K", 0, (0, 0)),  # Col 4
            Tile("F", 0, (0, 0)),  # Col 5
            Tile("M", 3, (0, 0)),  # Col 6
            Tile("D", 0, (0, 0)),  # Col 7
            Tile("G", 1, (0, 0)),              # Col 8
            Tile("H", 0, (0, 0)),              # Col 9
            None,              # Col 10
            None,              # Col 11
        ],

        # FILA 6
        [
            Tile("V", 2, (0, 0)),              # Col 0
            Tile("U", 1, (0, 0)),  # Col 1
            Tile("W", 2, (0, 0)),              # Col 2
            Tile("D", 1, (0, 0)),  # Col 3
            None,  # Col 4
            None,  # Col 5
            Tile("I", 3, (0, 0)),  # Col 6
            Tile("M", 3, (0, 0)),  # Col 7
            Tile("A", 1, (0, 0)),              # Col 8
            Tile("V", 3, (0, 0)),              # Col 9
            Tile("V", 0, (0, 0)),              # Col 10
            Tile("U", 0, (0, 0)),              # Col 11
        ],

        # FILA 7
        [
            None,              # Col 0
            Tile("B", 0, (0, 0)),  # Col 1
            Tile("E", 1, (0, 0)),  # Col 2
            Tile("R", 3, (0, 0)),  # Col 3
            Tile("E", 0, (0, 0)),  # Col 4
            Tile("Q", 3, (0, 0)),  # Col 5 - Con meeple centro
            Tile("N", 3, (0, 0)),  # Col 6
            Tile("R", 0, (0, 0)),              # Col 7
            None,              # Col 8
            Tile("D", 2, (0, 0)),              # Col 9
            None,              # Col 10
            None,              # Col 11
        ],

        # FILA 8
        [
            None,              # Col 0
            None,              # Col 1
            Tile("B", 0, (0, 0)),  # Col 2
            Tile("E", 0, (0, 0)),  # Col 3
            Tile("B", 0, (0, 0)),              # Col 4
            None,  # Col 5
            Tile("F", 1, (0, 0)),  # Col 6
            None,  # Col 7
            None,              # Col 8
            Tile("V", 0, (0, 0)),              # Col 9
            Tile("X", 1, (0, 0)),              # Col 10
            None,              # Col 11
        ],

        # FILA 9
        [
            None,              # Col 0
            None,              # Col 1
            None,  # Col 2
            None,  # Col 3
            None,  # Col 4
            None,  # Col 5 - Con meeple azul
            Tile("L", 3, (0, 0)),  # Col 6
            Tile("P", 3, (0, 0)),  # Col 7
            Tile("O", 3, (0, 0)),              # Col 8
            None,              # Col 9
            None,              # Col 10
            None,              # Col 11
        ],

        # FILA 10 (última)
        [
            None,              # Col 0
            None,              # Col 1
            None,  # Col 2
            None,  # Col 3
            None,  # Col 4
            None,  # Col 5 - Con meeple azul
            Tile("U", 0, (0, 0)),  # Col 6
            None,  # Col 7
            Tile("L", 3, (0, 0)),              # Col 8
            None,              # Col 9
            None,              # Col 10
            None,              # Col 11
        ]
    ]

    board = Board(tiles=ejemplo_tiles)
    
    # Configurar generador
    generator2 = BoardImageGenerator(
        tiles_folder="./tiles_texture_pack",  # Carpeta con archivos PNG/JPG
        tile_size=200            # Tamaño normalizado
    )
    
    generator = BoardImageGenerator(
        tiles_folder="./tiles",  # Carpeta con archivos PNG/JPG
        tile_size=200            # Tamaño normalizado
    )

    print("\nGenerando imagen de tableross pa...")
    generator.generate_board_image(board, "tablero_default_tiles.jpg")
    generator2.generate_board_image(board,"tablero_texture_pack.jpg")
    
    print("\n" + "=" * 60)
    print("\nCOLORES DE JUGADORES:")
    print("  Jugador 1: Rojo")
    print("  Jugador 2: Azul")
    print("=" * 60)