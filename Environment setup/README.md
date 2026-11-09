# Environment setup

Configuración para recrear el entorno **ENV_RL** en **Windows de 64 bits**, con **Python 3.10**, Jupyter en VS Code y Stable-Baselines3 en CPU.

Configuración acordada el **9 de noviembre de 2026**. La instalación de cada equipo se comprueba con el script y el notebook de esta carpeta.

## Qué contiene esta carpeta

| Archivo | Para qué sirve |
| --- | --- |
| [environment.yml](environment.yml) | Crear el entorno con las versiones acordadas. |
| [verificar_setup.py](verificar_setup.py) | Comprobar versiones, NumPy/PyTorch y un entrenamiento breve de PPO. |
| [prueba_interactiva.ipynb](prueba_interactiva.ipynb) | Probar el kernel de Jupyter, cálculos, gráficas y RL. |

## Versiones acordadas

| Componente | Versión |
| --- | --- |
| Python | 3.10.x |
| NumPy | 1.26.4 |
| SciPy | 1.14.1 |
| pandas | 2.2.3 |
| meshio | 5.3.5 |
| python-control, paquete control | 0.10.1 |
| PyTorch para CPU | 2.9.1 |
| Gymnasium | 1.0.0 |
| Stable-Baselines3 | 2.9.2 |

Matplotlib e ipykernel se resuelven a versiones compatibles al crear el entorno. El archivo fija las versiones principales; para conservar también las versiones de todas las dependencias, exporta el entorno después de validarlo.

Conda instala desde **conda-forge**. La única excepción es **Stable-Baselines3 2.9.2**, que se instala con pip como último paso automático: su distribución de PyPI admite Python 3.10, mientras que el paquete 2.9.2 de conda-forge exige Python 3.11. Sus dependencias ya se incluyen en la parte de Conda.

## 1. Preparar el equipo y obtener los archivos

Necesitas Miniconda, VS Code y las extensiones **Python** y **Jupyter**, publicadas por Microsoft.

Puedes obtener este repositorio sin instalar Git: en GitHub, pulsa **Code → Download ZIP** y descomprime el archivo. Si ya tienes una copia local del repositorio, utiliza esa carpeta.

Los comandos de Conda, Python y Git que aparecen aquí funcionan tanto en **CMD** como en **PowerShell**. Ejecútalos desde la raíz del repositorio: la carpeta que contiene CartPole, Stuart_Landau y Environment setup. Las rutas llevan comillas porque Environment setup contiene un espacio.

**Identifica tu terminal antes de cambiar de carpeta:**

- **CMD / Anaconda Prompt:** el prompt tiene una forma como (base) C:\...>.
- **PowerShell:** el prompt incluye PS, por ejemplo (base) PS C:\...>.

En **CMD**, usa este comando y sustituye la ruta por la ubicación de tu repositorio:

~~~cmd
cd /d "C:\ruta\FlowControlRL"
~~~

En **PowerShell**, el equivalente es:

~~~powershell
Set-Location "C:\ruta\FlowControlRL"
~~~

Set-Location, Remove-Item y los bloques de PowerShell no funcionan en CMD. El mensaje «Set-Location no se reconoce» indica que debes usar el comando cd /d de CMD.

Si Conda no se reconoce en la terminal de VS Code, abre **Anaconda Prompt** (CMD) o **Anaconda PowerShell Prompt** desde Inicio de Windows y sitúate en la carpeta del repositorio con el comando correspondiente. Comprueba:

~~~powershell
conda --version
~~~

## 2. Crear o activar el entorno

Si todavía tienes activo el antiguo entorno virtual creado con venv, ejecuta primero:

~~~powershell
deactivate
~~~

Para crear ENV_RL por primera vez:

~~~powershell
conda env create -f "Environment setup\environment.yml"
~~~

Cuando termine:

~~~powershell
conda activate ENV_RL
~~~

Si ENV_RL ya existe y funciona, basta con activarlo y pasar a la verificación. La creación no se repite cada vez que abres VS Code.

Comprueba Python y la ubicación del entorno:

~~~powershell
python -c "import sys; print(sys.version); print(sys.executable)"
conda env list
~~~

Debe aparecer Python 3.10.x. Este ENV_RL es un entorno **Conda**, guardado en la ubicación que muestra conda env list. La carpeta del repositorio guarda el código y esta configuración. El antiguo venv local, si lo conservas como respaldo, es un entorno distinto.

## 3. Configurar VS Code y Jupyter

En VS Code, abre la carpeta del repositorio.

1. Presiona **Ctrl + Shift + P → Python: Select Interpreter**.
2. Selecciona el ENV_RL de Conda. Si debes introducir una ruta, usa el ejecutable mostrado en el paso anterior; termina en **ENV_RL\python.exe**.
3. Desde una terminal con ENV_RL activo, registra el kernel:

~~~powershell
python -m ipykernel install --user --name env_rl --display-name "Python 3.10 (ENV_RL - CPU)"
~~~

Abre **prueba_interactiva.ipynb**, pulsa **Select Kernel / Seleccionar kernel** y elige **Python 3.10 (ENV_RL - CPU)**. Ejecuta las celdas en orden con **Shift + Enter**.

El primer bloque muestra el ejecutable y exige Python 3.10. El notebook comprueba también las versiones fijadas antes de ejecutar los ejemplos.

## 4. Verificar la instalación

Con ENV_RL activo, ejecuta desde la raíz del repositorio:

~~~powershell
python -m pip check
python "Environment setup\verificar_setup.py"
~~~

Debes obtener **No broken requirements found** y finalmente:

~~~text
PRUEBA COMPLETADA: versiones y entrenamiento en CPU correctos.
~~~

La prueba importa las librerías, comprueba sus versiones y ejecuta PPO durante 128 pasos en CartPole. Es una prueba de instalación; ese entrenamiento breve no mide la calidad de un agente.

En este setup, **torch.version.cuda es None**: corresponde a la compilación para CPU.

## 5. Volver a trabajar otro día

Abre el repositorio en VS Code y comprueba que el intérprete y el kernel siguen siendo ENV_RL.

En una terminal nueva que no lo tenga activo:

~~~powershell
conda activate ENV_RL
~~~

No hace falta reinstalar las librerías. Guarda los archivos antes de cerrar. Si se cierra o reinicia el kernel de Jupyter, vuelve a ejecutar sus celdas para recuperar las variables.

Para salir del entorno en la terminal:

~~~powershell
conda deactivate
~~~

## 6. Guardar la configuración realmente instalada

Después de superar las pruebas, exporta las versiones resueltas. Elige el comando de tu terminal; ambos eliminan la ruta local del archivo exportado.

**PowerShell:**

~~~powershell
conda env export --no-builds | Where-Object { $_ -notmatch '^prefix:' } | Set-Content -Encoding utf8 "Environment setup\environment_resuelto.yml"
~~~

**CMD / Anaconda Prompt:**

~~~cmd
conda env export --no-builds | findstr /V /B "prefix:" > "Environment setup\environment_resuelto.yml"
~~~

Guarda ese archivo en GitHub si quieres registrar las versiones de todas las dependencias. Para recrear esa configuración en otro equipo Windows compatible, usa ese archivo en lugar de environment.yml:

~~~powershell
conda env create -f "Environment setup\environment_resuelto.yml"
~~~

El entorno no debe existir todavía con el mismo nombre. Si quieres probar la recreación sin reemplazar el actual:

~~~powershell
conda env create --name ENV_RL_PRUEBA -f "Environment setup\environment_resuelto.yml"
conda activate ENV_RL_PRUEBA
~~~

Al usar otro nombre, selecciona también su intérprete y registra su kernel con un nombre diferente. Los paquetes binarios de un export de Windows pueden requerir ajustes al cambiar de sistema operativo.

## 7. Llevar el registro en GitHub

GitHub conserva las versiones de estos archivos mediante **commits**: cada commit es una entrada del historial con una descripción del cambio.

Para un cambio sencillo desde la web:

1. Abre environment.yml o README.md y pulsa el lápiz **Edit**.
2. Cambia las versiones o las instrucciones.
3. Pulsa **Commit changes** y escribe una descripción, por ejemplo: "Documentar nueva versión de SciPy".
4. Usa **History** para consultar los cambios anteriores.

Para añadir un export local, abre esta carpeta en GitHub y usa **Add file → Upload files**. Sube environment_resuelto.yml después de validarlo. Conserva las versiones principales solicitadas al añadir nuevas librerías y comprueba de nuevo la instalación.

Los archivos de configuración permiten reconstruir el entorno. La carpeta instalada de ENV_RL no necesita subirse a GitHub.

## Problemas que ya aparecieron durante la configuración

**VS Code envía el nombre del script con comillas adicionales.** El intérprete arranca, pero Python intenta abrir una ruta que contiene comillas literales. Como comprobación, ejecuta manualmente:

~~~powershell
python "Environment setup\verificar_setup.py"
~~~

Si funciona manualmente y falla desde el botón, revisa la extensión **Python Environments**. Deshabilitarla únicamente para este espacio de trabajo resolvió ese patrón en el setup inicial; conserva habilitadas las extensiones **Python** y **Jupyter** y vuelve a seleccionar el intérprete de Conda.

**El notebook usa otro Python.** Revisa Select Kernel y el ejecutable mostrado en la primera celda.

**Aparecen dos entornos en el prompt.** Comprueba siempre sys.executable. Para este proyecto usa el ENV_RL de Conda y desactiva el venv anterior antes de activarlo.

## Referencias de compatibilidad

- [NumPy 1.26.4: soporte para Python 3.9 a 3.12](https://numpy.org/doc/1.26/release/1.26.4-notes.html).
- [Stable-Baselines3 2.9.2 en PyPI](https://pypi.org/project/stable-baselines3/2.9.2/).
- [Metadatos de Stable-Baselines3 2.9.2 en conda-forge](https://api.anaconda.org/release/conda-forge/stable-baselines3/2.9.2).
- [Paquetes PyTorch CPU 2.9.1 de conda-forge](https://anaconda.org/conda-forge/pytorch-cpu/files?version=2.9.1).
