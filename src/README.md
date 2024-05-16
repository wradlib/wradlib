# Installation

## Prerequisites

In order to run the DjangoRadlib application, you need to have a Python interpreter installed on your local computer, as well as [a number of Python packages](#dependencies). We recommend installing [Anaconda](https://www.anaconda.com/download) as it includes Python, numerous required packages, and other useful tools (e.g., [Spyder](https://www.spyder-ide.org/)).

## Setting Up the Environment

Using Anaconda, the installation process is harmonized across platforms. Download and install the latest [Anaconda distribution](https://www.anaconda.com/download) for your specific OS. We recommend using the minimal distributions [Miniconda](https://conda.io/miniconda.html) or [Miniforge/Mambaforge](https://github.com/conda-forge/miniforge) if you do not want to install a full scientific Python stack.

We are constantly performing tests with the [conda-forge](https://conda-forge.org/) community channel (for the most recent three Python versions).

## Installation Steps

### Verify Python Installation

If your Python installation is working, the following command (in a console) should work:

```bash
$ python --version
Python 3.11.0
```

### Configure Conda

1. Add the conda-forge channel, where `wradlib` and its dependencies are located. Read more about the community effort [conda-forge](https://conda-forge.org):

    ```bash
    $ conda config --add channels conda-forge
    ```

2. Use strict channel priority to prevent channel clashes:

    ```bash
    $ conda config --set channel_priority strict
    ```

### Create and Activate Environment

3. Create a new environment from scratch:

    ```bash
    $ conda create --name djangoradlib python=3.11
    ```

4. Activate the `djangoradlib` environment:

    ```bash
    $ conda activate djangoradlib
    ```

### Install Dependencies

5. Install `wradlib`, Django, and other required packages specified in `requirements.txt`:

    Create a `requirements.txt` file with the following content:
    ```
    wradlib
    django
    django-crispy-forms==1.14.0
    pillow
    xhtml2pdf
    ```

    Then run:
    ```bash
    (djangoradlib) $ pip install -r requirements.txt
    ```

### Verify wradlib Installation

Test the integrity of your `wradlib` installation by opening a console window and calling the Python interpreter:

```bash
$ python
Python 3.11.0 | packaged by conda-forge | (main, Oct 25 2022, 06:24:40) [GCC 10.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

The Python prompt should appear. Then type:

```python
>>> import wradlib
>>> wradlib.__version__
'2.0.0'
```

If everything is okay, this will show the running `wradlib` version.

### Setting Up the Django Project

1. Clone the repository:

    ```sh
    git clone https://github.com/your-username/DjangoRadlib.git
    cd DjangoRadlib
    ```

2. Install the project dependencies:

    ```sh
    (djangoradlib) $ pip install -r requirements.txt
    ```

3. Set up the database:

    ```sh
    (djangoradlib) $ python manage.py migrate
    ```

4. Run the development server:

    ```sh
    (djangoradlib) $ python manage.py runserver
    ```

5. Access the application:
    Open your web browser and go to `http://127.0.0.1:8000`.
