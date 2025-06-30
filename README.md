# Dog Breed Recognition App

Este proyecto implementa un sistema completo para la **detección y clasificación de razas de perros**, combinando detección de objetos con YOLOv8 y clasificación de imágenes mediante redes convolucionales (**ResNet18** y una **CNN personalizada**). La aplicación incluye una interfaz interactiva en **Gradio**.

### Funcionalidades principales

- Entrenamiento de clasificadores personalizados y preentrenados  
- Búsqueda por similitud visual usando FAISS  
- Detección de perros con YOLOv8  
- Clasificación automática de razas en imágenes reales  
- Evaluación cuantitativa del pipeline completo (mAP, IoU, F1)  
- Anotación automática en formato YOLOv5 y COCO  
- Exportación a ONNX + aceleración con TensorRT

---

## Requisitos

- Python 3.10 o superior  
- `pip`  
- GPU NVIDIA con soporte CUDA 12.9 (recomendado para entrenamiento)  
- Sistema operativo: Linux


### Dependencias principales

Instalación con pip:

```bash
pip install -r requirements.txt
```

## Instrucciones de Ejecución

1. **Clonar el repositorio**
   
   ```bash
   git clone https://github.com/FrancoAntuna/CV-Dog-Breeds
   cd CV-Dog-Breeds
    ```
   #### Estructura esperada
   ├── coco/
   
   ├── embeddings/
   
   │   ├── embedding_matrix.pkl
   
   │   ├── index_custom.faiss
   
   │   ├── index_resnet.faiss
   
   │   ├── train_embeds_custom.npy
   
   │   ├── train_emneds_resnet.npy
   
   │   ├── train_labels_custom.npy
   
   │   └── train_labels_resnet.npy
   
   ├── labels_yolo/
   
   ├── modelos/
   
   │   ├── custom_cnn_dogbreeds.pth
   
   │   ├── dog_classifier_resnet18.onnx
   
   │   ├── resnet18_finetuned_dogbreeds.pth
   
   │   └── yolov8s.pt
   
   ├── cnn_custom.ipynb (este archivo es una base de pruebas para distintas cnn)
   
   ├── dog_breeds.ipynb (codigo fuente)
   
   ├── dogs.csv 


2. **Ubicar los modelos preentrenados**
    
    - Asegurarse de tener los siguientes archivos en la carpeta modelos/:
      - resnet18_finetuned_dogbreeds.pth
      - custom_cnn_dogbreeds.pth
    - Si no estan, se deben entrenar los modelos en la Etapa 2 o volve a clonar el git.

3. **Ejecutar el notebook principal**

    Abrir y correr el archivo:
     ```bash
        dog_breeds.ipynb
     ```
    Dentro del notebook se encuentra:

      - Descarga del dataset
      - Preparación del entorno
      - Etapas 1 a 4 del pipeline

4. **Ejecutar las primeras dos partes**
   
    Con esto se cargan las dependencias para la ejecucion modular de todas las etapas del proyecto.

6. **Consideraciones**
   
    La Etapa 1 recalcula los embeddings desde cero. Si ya tenés los archivos .npy generados, podés saltearla.
    La Etapa 2 vuelve a entrenar ambos modelos, lo cual puede tardar varios minutos.
    Asegurate de tener todas las dependencias correctamente instaladas (requirements.txt).


Este proyecto fue desarrollado como parte del curso de Visión por Computadora.
Para dudas, sugerencias o contribuciones, abrí un issue o enviá un Pull Request.
