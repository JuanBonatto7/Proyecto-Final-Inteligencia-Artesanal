"""
Herramienta de anotación adaptada para losetas con letras (A-X + blanco)
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from tile_mapping import TileMapper


class LetterTileAnnotationTool:
    """Herramienta GUI para anotar losetas usando letras"""
    
    def __init__(self, tiles_dir: str, reference_tiles_dir: str):
        self.tiles_dir = Path(tiles_dir)
        self.reference_dir = Path(reference_tiles_dir)
        self.mapper = TileMapper()
        
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
        """Carga las losetas de referencia"""
        refs = {}
        for idx in range(self.mapper.get_num_types()):
            ref_path = self.reference_dir / f'tile_type_{idx}.png'
            if ref_path.exists():
                img = cv2.imread(str(ref_path))
                letter = self.mapper.idx_to_letter(idx)
                refs[letter] = img
        return refs
    
    def setup_gui(self):
        """Configura la interfaz gráfica"""
        self.root = tk.Tk()
        self.root.title("Anotador de Losetas Carcassonne (Letras)")
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Imagen actual
        self.img_label = ttk.Label(main_frame)
        self.img_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Nombre archivo
        self.filename_label = ttk.Label(main_frame, text="", font=('Arial', 12, 'bold'))
        self.filename_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Tipo de loseta (usando letras)
        ttk.Label(main_frame, text="Tipo de loseta (letra):").grid(row=2, column=0, sticky=tk.W)
        self.tile_letter_var = tk.StringVar(value="A")
        
        # ComboBox con todas las letras disponibles
        letter_values = self.mapper.get_all_letters()
        tile_letter_combo = ttk.Combobox(main_frame, textvariable=self.tile_letter_var,
                                         values=letter_values, width=15, state='readonly')
        tile_letter_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Botón para ver referencia
        ttk.Button(main_frame, text="Ver referencia", 
                  command=self.show_current_reference).grid(row=2, column=2, padx=5)
        
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
        btn_frame.grid(row=7, column=0, columnspan=3, pady=20)
        
        ttk.Button(btn_frame, text="← Anterior", command=self.prev_tile).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Guardar", command=self.save_annotation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Siguiente →", command=self.next_tile).pack(side=tk.LEFT, padx=5)
        
        # Contador
        self.counter_label = ttk.Label(main_frame, text="", font=('Arial', 10))
        self.counter_label.grid(row=8, column=0, columnspan=3)
        
        # Botones adicionales
        btn_frame2 = ttk.Frame(main_frame)
        btn_frame2.grid(row=9, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame2, text="Ver todas referencias", 
                  command=self.show_all_references).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame2, text="Exportar JSON", 
                  command=self.export_json).pack(side=tk.LEFT, padx=5)
        
        # Atajos de teclado
        self.root.bind('<Left>', lambda e: self.prev_tile())
        self.root.bind('<Right>', lambda e: self.next_tile())
        self.root.bind('<Control-s>', lambda e: self.save_annotation())
        self.root.bind('<space>', lambda e: self.next_tile())
        
        # Atajos para letras (tecla rápida para seleccionar tipo)
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWX':
            self.root.bind(letter.lower(), lambda e, l=letter: self.quick_select_letter(l))
    
    def quick_select_letter(self, letter: str):
        """Selección rápida de letra con teclado"""
        if letter in self.mapper.get_all_letters():
            self.tile_letter_var.set(letter)
    
    def show_current_reference(self):
        """Muestra la loseta de referencia del tipo actual"""
        letter = self.tile_letter_var.get()
        if letter not in self.reference_tiles:
            messagebox.showwarning("Advertencia", f"No hay imagen de referencia para '{letter}'")
            return
        
        ref_img = self.reference_tiles[letter]
        
        # Crear ventana
        ref_window = tk.Toplevel(self.root)
        ref_window.title(f"Referencia: {letter}")
        
        # Convertir a PIL y mostrar
        img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img_pil)
        
        label = ttk.Label(ref_window, image=photo)
        label.image = photo
        label.pack(padx=20, pady=20)
        
        ttk.Label(ref_window, text=f"Tipo: {letter} (índice {self.mapper.letter_to_idx(letter)})",
                 font=('Arial', 12, 'bold')).pack(pady=10)
    
    def show_all_references(self):
        """Muestra ventana con todas las losetas de referencia"""
        ref_window = tk.Toplevel(self.root)
        ref_window.title("Losetas de Referencia")
        
        canvas = tk.Canvas(ref_window, width=900, height=700)
        scrollbar = ttk.Scrollbar(ref_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mostrar referencias en grid (5 por fila)
        for i, letter in enumerate(self.mapper.get_all_letters()):
            row = i // 5
            col = i % 5
            
            frame = ttk.Frame(scrollable_frame, relief='solid', borderwidth=1)
            frame.grid(row=row, column=col, padx=5, pady=5)
            
            if letter in self.reference_tiles:
                ref_img = self.reference_tiles[letter]
                img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(img_pil)
                
                label = ttk.Label(frame, image=photo)
                label.image = photo
                label.pack()
            else:
                ttk.Label(frame, text="No disponible", 
                         width=20, height=10).pack()
            
            ttk.Label(frame, text=f"{letter}", 
                     font=('Arial', 11, 'bold')).pack()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
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
                self.tile_letter_var.set(ann['tile_letter'])
                self.rotation_var.set(f"{ann['rotation']} ({ann['rotation']*90}°)")
                self.has_meeple_var.set(ann['has_meeple'])
                self.meeple_pos_var.set(str(ann['meeple_position']))
                self.meeple_color_var.set(ann['meeple_color'])
                self.toggle_meeple_position()
                return
        
        # Reset valores por defecto
        self.tile_letter_var.set("A")
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
        letter = self.tile_letter_var.get()
        
        annotation = {
            'image_path': tile_path,
            'tile_letter': letter,
            'tile_type': self.mapper.letter_to_idx(letter),  # También guardar índice
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
        
        messagebox.showinfo("Guardado", f"Anotación guardada: {letter}")
    
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
    
    def export_json(self):
        """Exporta anotaciones a JSON"""
        output_file = 'annotations_with_letters.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        messagebox.showinfo("Exportado", 
                           f"Anotaciones guardadas en {output_file}\n"
                           f"Total: {len(self.annotations)} losetas anotadas")
    
    def run(self):
        """Ejecuta la aplicación"""
        self.root.mainloop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python annotation_tool_letters.py <tiles_dir> <references_dir>")
        print("\nEjemplo:")
        print("  python annotation_tool_letters.py ./tiles ./referencias")
        print("\nAsegúrate de haber ejecutado primero:")
        print("  python tile_mapping.py prepare ./letras ./referencias")
        sys.exit(1)
    
    tool = LetterTileAnnotationTool(sys.argv[1], sys.argv[2])
    tool.run()
