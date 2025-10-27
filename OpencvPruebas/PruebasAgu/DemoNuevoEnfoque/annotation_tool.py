import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


class TileAnnotationTool:
    """Herramienta GUI para anotar losetas de Carcassonne"""
    
    def __init__(self, tiles_dir: str, reference_tiles_dir: str):
        self.tiles_dir = Path(tiles_dir)
        self.reference_dir = Path(reference_tiles_dir)
        
        # Cargar imágenes de referencia
        self.reference_tiles = self.load_reference_tiles()
        
        # Cargar tiles a anotar
        self.tile_files = sorted(list(self.tiles_dir.glob('*.png')))
        self.current_idx = 0
        
        # Anotaciones
        self.annotations = []
        
        # Setup GUI
        self.setup_gui()
        self.load_current_tile()
    
    def load_reference_tiles(self):
        """Carga las 24 losetas de referencia"""
        refs = []
        for i in range(24):
            ref_path = self.reference_dir / f'tile_type_{i}.png'
            if ref_path.exists():
                refs.append(cv2.imread(str(ref_path)))
            else:
                refs.append(None)
        return refs
    
    def setup_gui(self):
        """Configura la interfaz gráfica"""
        self.root = tk.Tk()
        self.root.title("Anotador de Losetas Carcassonne")
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Imagen actual
        self.img_label = ttk.Label(main_frame)
        self.img_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Nombre archivo
        self.filename_label = ttk.Label(main_frame, text="", font=('Arial', 12, 'bold'))
        self.filename_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Tipo de loseta
        ttk.Label(main_frame, text="Tipo de loseta (0-23):").grid(row=2, column=0, sticky=tk.W)
        self.tile_type_var = tk.StringVar(value="0")
        tile_type_spin = ttk.Spinbox(main_frame, from_=0, to=23, textvariable=self.tile_type_var, width=10)
        tile_type_spin.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Rotación
        ttk.Label(main_frame, text="Rotación:").grid(row=3, column=0, sticky=tk.W)
        self.rotation_var = tk.StringVar(value="0")
        rotation_combo = ttk.Combobox(main_frame, textvariable=self.rotation_var, 
                                      values=['0 (0°)', '1 (90°)', '2 (180°)', '3 (270°)'],
                                      width=15, state='readonly')
        rotation_combo.current(0)
        rotation_combo.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Tiene ficha
        self.has_meeple_var = tk.BooleanVar()
        meeple_check = ttk.Checkbutton(main_frame, text="Tiene ficha de jugador",
                                       variable=self.has_meeple_var,
                                       command=self.toggle_meeple_position)
        meeple_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Posición de ficha
        ttk.Label(main_frame, text="Posición ficha (0-8):").grid(row=5, column=0, sticky=tk.W)
        self.meeple_pos_var = tk.StringVar(value="-1")
        self.meeple_pos_spin = ttk.Spinbox(main_frame, from_=-1, to=8, 
                                           textvariable=self.meeple_pos_var, width=10)
        self.meeple_pos_spin.grid(row=5, column=1, sticky=tk.W, pady=5)
        self.meeple_pos_spin.config(state='disabled')
        
        # Color de ficha
        ttk.Label(main_frame, text="Color de ficha:").grid(row=6, column=0, sticky=tk.W)
        self.meeple_color_var = tk.StringVar(value="none")
        self.meeple_color_combo = ttk.Combobox(main_frame, textvariable=self.meeple_color_var,
                                               values=['none', 'red', 'blue', 'green', 'yellow', 'black'],
                                               width=15, state='readonly')
        self.meeple_color_combo.current(0)
        self.meeple_color_combo.grid(row=6, column=1, sticky=tk.W, pady=5)
        self.meeple_color_combo.config(state='disabled')
        
        # Botones de navegación
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="← Anterior", command=self.prev_tile).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Guardar", command=self.save_annotation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Siguiente →", command=self.next_tile).pack(side=tk.LEFT, padx=5)
        
        # Contador
        self.counter_label = ttk.Label(main_frame, text="", font=('Arial', 10))
        self.counter_label.grid(row=8, column=0, columnspan=2)
        
        # Botones adicionales
        btn_frame2 = ttk.Frame(main_frame)
        btn_frame2.grid(row=9, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame2, text="Ver referencias", command=self.show_references).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame2, text="Exportar JSON", command=self.export_json).pack(side=tk.LEFT, padx=5)
        
        # Atajos de teclado
        self.root.bind('<Left>', lambda e: self.prev_tile())
        self.root.bind('<Right>', lambda e: self.next_tile())
        self.root.bind('<Control-s>', lambda e: self.save_annotation())
        self.root.bind('<space>', lambda e: self.next_tile())
    
    def toggle_meeple_position(self):
        """Habilita/deshabilita campos de ficha"""
        if self.has_meeple_var.get():
            self.meeple_pos_spin.config(state='normal')
            self.meeple_color_combo.config(state='readonly')
            self.meeple_pos_var.set("0")
            self.meeple_color_var.set("red")
        else:
            self.meeple_pos_spin.config(state='disabled')
            self.meeple_color_combo.config(state='disabled')
            self.meeple_pos_var.set("-1")
            self.meeple_color_var.set("none")
    
    def load_current_tile(self):
        """Carga la loseta actual"""
        if self.current_idx >= len(self.tile_files):
            messagebox.showinfo("Completado", "¡Has anotado todas las losetas!")
            return
        
        tile_path = self.tile_files[self.current_idx]
        
        # Cargar imagen
        img = Image.open(tile_path)
        img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img)
        
        self.img_label.config(image=photo)
        self.img_label.image = photo
        
        # Actualizar labels
        self.filename_label.config(text=tile_path.name)
        self.counter_label.config(text=f"Loseta {self.current_idx + 1} de {len(self.tile_files)}")
        
        # Cargar anotación previa si existe
        self.load_previous_annotation()
    
    def load_previous_annotation(self):
        """Carga anotación previa si existe"""
        tile_path = str(self.tile_files[self.current_idx])
        
        for ann in self.annotations:
            if ann['image_path'] == tile_path:
                self.tile_type_var.set(str(ann['tile_type']))
                self.rotation_var.set(f"{ann['rotation']} ({ann['rotation']*90}°)")
                self.has_meeple_var.set(ann['has_meeple'])
                self.meeple_pos_var.set(str(ann['meeple_position']))
                self.meeple_color_var.set(ann['meeple_color'])
                self.toggle_meeple_position()
                return
        
        # Reset valores por defecto
        self.tile_type_var.set("0")
        self.rotation_var.set("0 (0°)")
        self.has_meeple_var.set(False)
        self.meeple_pos_var.set("-1")
        self.meeple_color_var.set("none")
        self.toggle_meeple_position()
    
    def save_annotation(self):
        """Guarda la anotación actual"""
        tile_path = str(self.tile_files[self.current_idx])
        
        # Extraer rotación
        rotation = int(self.rotation_var.get().split()[0])
        
        annotation = {
            'image_path': tile_path,
            'tile_type': int(self.tile_type_var.get()),
            'rotation': rotation,
            'has_meeple': self.has_meeple_var.get(),
            'meeple_position': int(self.meeple_pos_var.get()),
            'meeple_color': self.meeple_color_var.get()
        }
        
        # Actualizar o agregar
        found = False
        for i, ann in enumerate(self.annotations):
            if ann['image_path'] == tile_path:
                self.annotations[i] = annotation
                found = True
                break
        
        if not found:
            self.annotations.append(annotation)
        
        messagebox.showinfo("Guardado", "Anotación guardada correctamente")
    
    def next_tile(self):
        """Va a la siguiente loseta"""
        if self.current_idx < len(self.tile_files) - 1:
            self.current_idx += 1
            self.load_current_tile()
    
    def prev_tile(self):
        """Va a la loseta anterior"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_tile()
    
    def show_references(self):
        """Muestra ventana con losetas de referencia"""
        ref_window = tk.Toplevel(self.root)
        ref_window.title("Losetas de Referencia")
        
        canvas = tk.Canvas(ref_window, width=800, height=600)
        scrollbar = ttk.Scrollbar(ref_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mostrar referencias en grid
        for i, ref_img in enumerate(self.reference_tiles):
            if ref_img is not None:
                row = i // 4
                col = i % 4
                
                img = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
                img.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(img)
                
                label = ttk.Label(scrollable_frame, image=photo)
                label.image = photo
                label.grid(row=row*2, column=col, padx=5, pady=5)
                
                ttk.Label(scrollable_frame, text=f"Tipo {i}").grid(row=row*2+1, column=col)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def export_json(self):
        """Exporta anotaciones a JSON"""
        output_file = 'annotations.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        messagebox.showinfo("Exportado", f"Anotaciones guardadas en {output_file}")
    
    def run(self):
        """Ejecuta la aplicación"""
        self.root.mainloop()


# Script de línea de comandos más simple
class SimpleCLIAnnotator:
    """Anotador simple por línea de comandos"""
    
    def __init__(self, tiles_dir: str):
        self.tiles_dir = Path(tiles_dir)
        self.tile_files = sorted(list(self.tiles_dir.glob('*.png')))
        self.annotations = []
    
    def annotate(self):
        """Proceso de anotación"""
        print("=== ANOTADOR DE LOSETAS CARCASSONNE ===\n")
        
        for i, tile_path in enumerate(self.tile_files):
            print(f"\n[{i+1}/{len(self.tile_files)}] {tile_path.name}")
            
            # Mostrar imagen
            img = cv2.imread(str(tile_path))
            cv2.imshow('Loseta actual', img)
            cv2.waitKey(500)
            
            # Solicitar datos
            tile_type = int(input("Tipo de loseta (0-23): "))
            rotation = int(input("Rotación (0=0°, 1=90°, 2=180°, 3=270°): "))
            has_meeple = input("¿Tiene ficha? (s/n): ").lower() == 's'
            
            if has_meeple:
                meeple_pos = int(input("Posición de ficha (0-8): "))
                meeple_color = input("Color (red/blue/green/yellow/black): ")
            else:
                meeple_pos = -1
                meeple_color = 'none'
            
            annotation = {
                'image_path': str(tile_path),
                'tile_type': tile_type,
                'rotation': rotation,
                'has_meeple': has_meeple,
                'meeple_position': meeple_pos,
                'meeple_color': meeple_color
            }
            
            self.annotations.append(annotation)
            
            # Opción de retroceder
            if input("\n¿Continuar? (Enter=Sí, u=Deshacer): ") == 'u':
                self.annotations.pop()
                continue
        
        cv2.destroyAllWindows()
        
        # Guardar
        with open('annotations.json', 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"\n✓ Anotaciones guardadas en annotations.json")


def create_reference_tiles_from_models(models_dir: str, output_dir: str):
    """
    Crea el directorio de referencia con las 24 losetas modelo
    Las losetas modelo deben estar nombradas como: tipo_0.png, tipo_1.png, etc.
    """
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for i in range(24):
        # Buscar diferentes nombres posibles
        possible_names = [
            f'tipo_{i}.png', f'type_{i}.png', f'tile_{i}.png',
            f'modelo_{i}.png', f'model_{i}.png'
        ]
        
        for name in possible_names:
            model_file = models_path / name
            if model_file.exists():
                output_file = output_path / f'tile_type_{i}.png'
                img = cv2.imread(str(model_file))
                cv2.imwrite(str(output_file), img)
                print(f"✓ Tipo {i} copiado")
                break
        else:
            print(f"✗ Tipo {i} no encontrado")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python annotation_tool.py gui <tiles_dir> <references_dir>")
        print("  python annotation_tool.py cli <tiles_dir>")
        print("  python annotation_tool.py prepare <models_dir> <output_dir>")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'gui':
        if len(sys.argv) < 4:
            print("Error: Especifica tiles_dir y references_dir")
            sys.exit(1)
        
        tool = TileAnnotationTool(sys.argv[2], sys.argv[3])
        tool.run()
    
    elif mode == 'cli':
        if len(sys.argv) < 3:
            print("Error: Especifica tiles_dir")
            sys.exit(1)
        
        tool = SimpleCLIAnnotator(sys.argv[2])
        tool.annotate()
    
    elif mode == 'prepare':
        if len(sys.argv) < 4:
            print("Error: Especifica models_dir y output_dir")
            sys.exit(1)
        
        create_reference_tiles_from_models(sys.argv[2], sys.argv[3])
    
    else:
        print(f"Modo desconocido: {mode}")
