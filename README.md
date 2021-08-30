# Deteccion de objetos en video 
Proyecto de detección de objetos basado en [PyTorch YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) y realizado por @puigalex para correr detección de objetos sobre video. El objetivo es usar un entrenamiento personalizado para detectar copas y troncos de árboles.

[YOLO](https://pjreddie.com/darknet/yolo/) (**You Only Look Once**) es un modelo el cual esta optimizado para generar detecciones de elementos a una velocidad muy alta, es por eso que es una muy buena opción para usarlo en video. Tanto el entrenamiento como predicciones con este modelo se ven beneficiadas si se cumple con una computadora que tenga una GPU NVIDIA.

Por defecto este modelo esta pre entrenado para detecta 80 distintos objetos, la lista de estos se encuentra en el archivo [data/coco.names](https://github.com/AlbertoPerezGant/sifelec/blob/master/data/coco.names). Este se puede usar como prueba para evaluar la detección básica.

Los pasos a seguir para poder correr detección de objetos en el video de una webcam son los siguientes:

# Crear ambiente
Para tener en orden nuestras paqueterias de python primero vamos a crear un ambiente llamado "deteccionobj" el cual tiene la version 3.6 de python. Para el entrenamiento de un nuevo modelo podemos usar "yolotrain" donde usamos la misma versión de python
``` 
conda create -n deteccionobj python=3.6
```

Activamos el ambiende 'deteccionobj' para asegurarnos que tenemos un entorno limpio en el que trabajar con las mínimas incompatibilidades entre dependencias. En el caso de instalar las dependencias en Windows, es posible encontrar error al instalar torch. 
```
conda activate deteccionobj
```

# Instalación de las paqueterias
Estando dentro de nuestro ambiente se instalan todas las paqueterias necesarias para correr nuestro detector de objetos en video, la lista de las paqueterias y versiones a instalar están dentro del archivo requirements.txt por lo cual instalaremos haciendo referencia a ese archivo. Es posible que librerias como tensorflow o torch den problemas si no se realiza en Linux debido a la versión que viene definida en requirements.txt. Es recomendable realizar este proceso en un sistema Linux. Yo lo he realizado en WSL2 con Ubuntu.
```
pip install -r requirements.txt
```

# Descargar los pesos del modelo entrenado 
Para poder correr el modelo de yolo tendremos que descargar los pesos de la red neuronal, los pesos son los valores que tienen todas las conexiones entre las neuronas de la red neuronal de YOLO, este tipo de modelos son computacionalmente muy pesados de entrenar desde cero por lo cual descargar el modelo pre entrenado es una buena opción.

```
cd weights
bash download_weights.sh
```

# Correr el detector de objetos en video 
Por ultimo corremos este comando el cual activa la camara web para poder hacer deteccion de video sobre un video "en vivo"
```
python deteccion_video.py
```

# Modificaciones
Si en vez de correr detección de objetos sobre la webcam lo que quieres es correr el modelo sobre un video que ya fue pre grabado tienes que cambiar el comando para correr el codigo a:

```
python deteccion_video.py --webcam 0 --directorio_video <directorio_al_video.mp4>
```

# Entrenamiento 

Ahora, si lo que quieres es entrenar un modelo con las clases que tu quieras y no utilizar las 80 clases que vienen por defecto podemos entrenar nuestro propio modelo. Estos son los pasos que deberás seguir:

Primero deberás etiquetar las imagenes con el formato VOC, aqui tengo un video explicando como hacer este etiquetado: 

Desde la carpeta config correremos el archivo create_custom_model para generar un archivo .cfg el cual contiene información sobre la red neuronal para correr las detecciones
```
cd config
bash create_custom_model.sh <Numero_de_clases_a_detectar>
cd ..
```
Descargamos la estructura de pesos de YOLO para poder hacer transfer learning sobre esos pesos
```
cd weights
bash download_darknet.sh
cd ..
```

## Poner las imagenes y archivos de metadata en las carpetar necesarias

Esto se puede hacer con el siguiente comando:

```
labelImg
```
Se abrirá una ventana con una aplicación donde se configura el metodo de salida a YOLO y se seleccionan las clases a definir en cada objeto. Si se está trabajando con WSL2, será necesario ejecutar el comando desde una terminal cmd ya que por el momento WSL2 no permite la ejecucion de aplicaciones GUI. El directorio de salida será **data/custom/labels** y etrada **data/custom/images**.

Las imagenes etiquetadas tienen que estar en el directorio **data/custom/images** mientras que las etiquetas/metadata de las imagenes tienen que estar en **data/custom/labels**.
Por cada imagen.jpg debe de existir un imagen.txt (metadata con el mismo nombre de la imagen)

El archivo ```data/custom/classes.names``` debe contener el nombre de las clases, como fueron etiquetadas, un renglon por clase.

Los archivos ```data/custom/valid.txt``` y ```data/custom/train.txt``` deben contener la dirección donde se encuentran cada una de las imagenes. Estos se pueden generar con el siguiente comando (estando las imagenes ya dentro de ```data/custom/images```)
```
python split_train_val.py
```

## Entrenar

 ```
 python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74 --batch_size 2
 ```

## Correr deteccion de objetos en video con nuestras clases
```
python deteccion_video.py --model_def config/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_99.pth --class_path data/custom/classes.names  --weights_path checkpoints/yolov3_ckpt_99.pth  --conf_thres 0.85
```
