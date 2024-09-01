import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
from PIL import Image, ImageTk
from app import segmentar_imagen
import os
import time

def cargar_carpeta():
    carpeta = filedialog.askdirectory()
    if carpeta:
        archivos_imagen = [os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]
        if archivos_imagen:
            boton_cargar.pack_forget()
            procesar_imagenes(archivos_imagen)
        else:
            messagebox.showinfo("Información", "No se encontraron imágenes en la carpeta seleccionada.")
    else:
        messagebox.showinfo("Información", "No se seleccionó ninguna carpeta.")

def actualizar_estado(mensaje):
    boton_estado.config(text=mensaje)

def procesar_imagenes(archivos_imagen):
    with open("donuts.csv", mode="w", newline="") as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow(["Donut", "Numero de Virutas", "Tiempo de Procesado", "Valido"])
        for id, ruta_imagen in enumerate(archivos_imagen):
            try:
                # Limpiar la pantalla de imágenes anteriores
                etiqueta_imagen_segmentada.place_forget()
                actualizar_estado("Cargando imagen...")
                boton_estado.config(bg="white")

                # Cargar la imagen original
                imagen = Image.open(ruta_imagen)
                imagen.thumbnail((300, 300))
                img_tk = ImageTk.PhotoImage(imagen)

                # Mostrar la imagen original en la animación
                animar_imagen(imagen, img_tk)

                # Procesar la imagen
                numero_virutas, area_total_virutas, tiempo_procesamiento, valido = procesar_imagen(ruta_imagen, imagen)
                escritor_csv.writerow([id, numero_virutas, tiempo_procesamiento, valido])
                # Mostrar los metadatos de la imagen
                mostrar_metadatos(numero_virutas, area_total_virutas, tiempo_procesamiento)

                # Retardo para animación
                ventana.update()
                time.sleep(5)

            except Exception as e:
                messagebox.showerror("Error", f"No se pudo procesar la imagen: {str(e)}")
                break

def animar_imagen(imagen, img_tk):
    etiqueta_imagen.place(x=-300, y=ventana.winfo_height() // 2 - 150)  # Posición inicial a la izquierda de la ventana
    etiqueta_imagen.config(image=img_tk)
    etiqueta_imagen.image = img_tk

    for x in range(-300, ventana.winfo_width() // 2 - 150, 10):  # Desplazamiento hacia el centro
        etiqueta_imagen.place(x=x, y=ventana.winfo_height() // 2 - 150)
        ventana.update()
        time.sleep(0.02)

    actualizar_estado("Procesando imagen...")
    boton_estado.config(bg="blue")
    ventana.update()
    etiqueta_imagen.place_forget()  # Ocultar la imagen original

def procesar_imagen(ruta_imagen, imagen):
    try:
        numero_virutas, area_total_virutas, tiempo_procesamiento, valido = segmentar_imagen(ruta_imagen)
        imagen = Image.open("imagen_segmentada.jpg")
        imagen.thumbnail((300, 300))
        # Determinar si la imagen segmentada es válida
        if valido:
            resultado_texto = "Imagen válida"
            color = "green"
        else:
            resultado_texto = "Imagen no válida"
            color = "red"
        
        # Alinear la imagen segmentada a la derecha del área coloreada
        x_pos = ventana.winfo_width() // 2 + 200
        y_pos = (ventana.winfo_height() // 2) - (imagen.size[1] // 2)  # Alinear verticalmente

        img_segmentada_tk = ImageTk.PhotoImage(imagen)
        etiqueta_imagen_segmentada.config(image=img_segmentada_tk)
        etiqueta_imagen_segmentada.image = img_segmentada_tk
        etiqueta_imagen_segmentada.place(x=x_pos, y=y_pos)

        # Mostrar el resultado de la validación debajo de la imagen procesada
        actualizar_estado(resultado_texto)
        boton_estado.config(bg=color)

        return numero_virutas, area_total_virutas, tiempo_procesamiento, valido

    except Exception as e:
        print(traceback.format_exc())
        messagebox.showerror("Error", f"No se pudo procesar la imagen: {str(e)}")

def mostrar_metadatos(numero_virutas, area_total_virutas, tiempo_procesamiento):
    etiqueta_tiempo.config(text=f"Tiempo de procesamiento: {tiempo_procesamiento}")
    etiqueta_virutas.config(text=f"Número de virutas: {numero_virutas}")
    etiqueta_area_total.config(text=f"Área total de las virutas: {area_total_virutas}")

# Configuración de la ventana principal
ventana = tk.Tk()
ventana.title("Inspección Visual de Imágenes")

ventana.geometry("800x600")  # Para un tamaño inicial de ventana
ventana.attributes('-zoomed', True)  # Maximizar ventana en sistemas compatibles

# Crear un marco para los botones (parte superior)
frame_botones = tk.Frame(ventana)
frame_botones.pack(side=tk.TOP, pady=10)

# Botón para cargar la carpeta de imágenes
boton_cargar = tk.Button(frame_botones, text="Cargar Carpeta", command=cargar_carpeta,  width=20, height=2, font=("Arial", 12))
boton_cargar.pack(side=tk.TOP, padx=10)

# Botón de estado
boton_estado = tk.Button(frame_botones, text="Estado", state=tk.DISABLED,  width=50, height=4, font=("Arial", 12))
boton_estado.pack(side=tk.BOTTOM, padx=10)

# Etiqueta para previsualizar la imagen original (animada)
etiqueta_imagen = tk.Label(ventana)

# Etiqueta para previsualizar la imagen segmentada (derecha)
etiqueta_imagen_segmentada = tk.Label(ventana)

# Crear un marco para los metadatos (parte inferior derecha)
frame_metadatos = tk.Frame(ventana)
frame_metadatos.pack(side=tk.BOTTOM, pady=10, fill=tk.X)

# Metadatos (en el marco inferior derecho) con texto más grande
etiqueta_tiempo = tk.Label(frame_metadatos, text="Tiempo de procesamiento: -", font=("Arial", 14))
etiqueta_tiempo.pack(side=tk.TOP)

etiqueta_virutas = tk.Label(frame_metadatos, text="Número de virutas: -", font=("Arial", 14))
etiqueta_virutas.pack(side=tk.BOTTOM)

etiqueta_area_total = tk.Label(frame_metadatos, text="Área total de las virutas: -", font=("Arial", 14))
etiqueta_area_total.pack(side=tk.BOTTOM)

# Iniciar la aplicación
ventana.mainloop()
