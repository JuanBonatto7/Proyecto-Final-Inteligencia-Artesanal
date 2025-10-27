"""
Sistema de mapeo entre letras de losetas y índices numéricos
Para Carcassonne con losetas nombradas como letras (A-X + blanco)
"""

class TileMapper:
    """Maneja la conversión entre letras y índices de losetas"""
    
    def __init__(self):
        # Crear mapeo: A=0, B=1, ..., X=23, blanco=24
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWX")
        self.letters.append("BLANCO")
        
        # Diccionarios de conversión
        self.letter_to_index = {letter: idx for idx, letter in enumerate(self.letters)}
        self.index_to_letter = {idx: letter for idx, letter in enumerate(self.letters)}
        
        # Total de tipos (24 letras + 1 blanco = 25)
        self.num_types = len(self.letters)
    
    def letter_to_idx(self, letter: str) -> int:
        """Convierte letra a índice numérico"""
        letter = letter.upper()
        if letter in self.letter_to_index:
            return self.letter_to_index[letter]
        raise ValueError(f"Letra no válida: {letter}")
    
    def idx_to_letter(self, idx: int) -> str:
        """Convierte índice a letra"""
        if idx in self.index_to_letter:
            return self.index_to_letter[idx]
        raise ValueError(f"Índice no válido: {idx}")
    
    def get_all_letters(self):
        """Retorna lista de todas las letras"""
        return self.letters.copy()
    
    def get_num_types(self) -> int:
        """Retorna número total de tipos de losetas"""
        return self.num_types
    
    def filename_to_idx(self, filename: str) -> int:
        """
        Convierte nombre de archivo a índice
        Ejemplos: 'A.jpg' -> 0, 'B.jpg' -> 1, 'blanco.jpg' -> 24
        """
        # Extraer la letra del nombre del archivo
        name = filename.split('.')[0].upper()
        return self.letter_to_idx(name)
    
    def idx_to_filename(self, idx: int, extension: str = 'jpg') -> str:
        """
        Convierte índice a nombre de archivo
        Ejemplo: 0 -> 'A.jpg', 24 -> 'blanco.jpg'
        """
        letter = self.idx_to_letter(idx)
        return f"{letter}.{extension}"


def prepare_reference_tiles(input_dir: str, output_dir: str):
    """
    Prepara las losetas de referencia desde el formato de letras
    al formato esperado por el sistema (tile_type_0.png, etc.)
    
    Args:
        input_dir: Directorio con A.jpg, B.jpg, ..., X.jpg, blanco.jpg
        output_dir: Directorio de salida con tile_type_0.png, etc.
    """
    import cv2
    from pathlib import Path
    
    mapper = TileMapper()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("PREPARANDO LOSETAS DE REFERENCIA")
    print("="*60)
    
    success_count = 0
    
    for letter in mapper.get_all_letters():
        # Buscar archivo de entrada
        possible_extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
        input_file = None
        
        for ext in possible_extensions:
            candidate = input_path / f"{letter}.{ext}"
            if candidate.exists():
                input_file = candidate
                break
        
        if input_file is None:
            print(f"⚠️  {letter}: No encontrado")
            continue
        
        # Leer imagen
        img = cv2.imread(str(input_file))
        if img is None:
            print(f"⚠️  {letter}: Error al leer")
            continue
        
        # Guardar con nuevo nombre
        idx = mapper.letter_to_idx(letter)
        output_file = output_path / f"tile_type_{idx}.png"
        cv2.imwrite(str(output_file), img)
        
        print(f"✓ {letter} -> tile_type_{idx}.png")
        success_count += 1
    
    print("\n" + "="*60)
    print(f"✓ COMPLETADO: {success_count}/{mapper.get_num_types()} losetas procesadas")
    print(f"Directorio de salida: {output_dir}")
    print("="*60)


def convert_annotations_to_letters(annotations_file: str, output_file: str):
    """
    Convierte un archivo de anotaciones con índices a formato legible con letras
    """
    import json
    
    mapper = TileMapper()
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Añadir campo 'tile_letter' a cada anotación
    for ann in annotations:
        ann['tile_letter'] = mapper.idx_to_letter(ann['tile_type'])
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Anotaciones convertidas guardadas en {output_file}")


def create_letter_based_annotation_template(output_file: str = 'annotations_letters_template.json'):
    """
    Crea plantilla de anotaciones usando letras en lugar de números
    """
    import json
    
    template = [
        {
            "image_path": "tiles/tile_000_r0_c0.png",
            "tile_letter": "A",  # Usa letra en lugar de número
            "rotation": 0,
            "has_meeple": False,
            "meeple_position": -1,
            "meeple_color": "none"
        }
    ]
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Plantilla creada en {output_file}")
    print("\nInstrucciones:")
    print("1. Duplica la estructura para cada loseta")
    print("2. tile_letter: A-X o 'blanco' (se convertirá automáticamente a índice)")
    print("3. rotation: 0=0°, 1=90°, 2=180°, 3=270°")
    print("4. has_meeple: true/false")
    print("5. meeple_position: 0-8 (posición en la loseta), -1 si no hay")
    print("6. meeple_color: 'red', 'blue', 'green', 'yellow', 'black', 'none'")


def convert_letter_annotations_to_numeric(input_file: str, output_file: str):
    """
    Convierte anotaciones con letras a formato numérico para entrenamiento
    """
    import json
    
    mapper = TileMapper()
    
    with open(input_file, 'r') as f:
        annotations = json.load(f)
    
    # Convertir tile_letter a tile_type
    for ann in annotations:
        if 'tile_letter' in ann:
            ann['tile_type'] = mapper.letter_to_idx(ann['tile_letter'])
        elif 'tile_type' not in ann:
            raise ValueError(f"Anotación sin tile_letter ni tile_type: {ann}")
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Anotaciones numéricas guardadas en {output_file}")
    print(f"Total: {len(annotations)} anotaciones")


def show_reference_tiles_map(references_dir: str):
    """
    Muestra un mapeo visual de todas las losetas de referencia
    """
    import cv2
    import numpy as np
    from pathlib import Path
    
    mapper = TileMapper()
    ref_path = Path(references_dir)
    
    print("\n" + "="*60)
    print("MAPEO DE LOSETAS DE REFERENCIA")
    print("="*60)
    
    for idx, letter in enumerate(mapper.get_all_letters()):
        ref_file = ref_path / f"tile_type_{idx}.png"
        
        status = "✓" if ref_file.exists() else "✗"
        print(f"{status} Índice {idx:2d} = Letra '{letter:6s}' -> {ref_file.name}")
    
    print("="*60)


# Uso desde línea de comandos
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python tile_mapping.py prepare <input_dir> <output_dir>")
        print("     Prepara losetas de referencia desde formato de letras")
        print()
        print("  python tile_mapping.py convert <input_annotations.json> <output.json>")
        print("     Convierte anotaciones con letras a formato numérico")
        print()
        print("  python tile_mapping.py template")
        print("     Crea plantilla de anotaciones con letras")
        print()
        print("  python tile_mapping.py show <references_dir>")
        print("     Muestra mapeo de losetas de referencia")
        print()
        print("Ejemplos:")
        print("  python tile_mapping.py prepare ./letras ./referencias")
        print("  python tile_mapping.py convert annotations_letras.json train_annotations.json")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'prepare':
        if len(sys.argv) < 4:
            print("Error: Especifica input_dir y output_dir")
            sys.exit(1)
        prepare_reference_tiles(sys.argv[2], sys.argv[3])
    
    elif command == 'convert':
        if len(sys.argv) < 4:
            print("Error: Especifica input y output file")
            sys.exit(1)
        convert_letter_annotations_to_numeric(sys.argv[2], sys.argv[3])
    
    elif command == 'template':
        create_letter_based_annotation_template()
    
    elif command == 'show':
        if len(sys.argv) < 3:
            print("Error: Especifica references_dir")
            sys.exit(1)
        show_reference_tiles_map(sys.argv[2])
    
    else:
        print(f"Comando desconocido: {command}")
