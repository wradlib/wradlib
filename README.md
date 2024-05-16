
## Djangoradlib: A Web Application for Radar Beam Blockage Simulation

### Overview

DjangoRadlib is a sophisticated web application designed to aid in radar site prospection by simulating radar beam blockage. Leveraging the capabilities of the `wradlib` library for radar data processing and the Django web framework, this application provides an interactive platform for assessing radar performance from any prospective site. The tool is ideal for meteorologists, hydrologists, and engineers involved in radar placement and planning, ensuring optimized radar deployment and accurate data collection.

### Features

- **Interactive Simulations**: Users can input site coordinates and radar parameters to run real-time simulations of beam blockage, visualizing the impact of terrain and obstructions on radar coverage.
- **Digital Elevation Model (DEM) Integration**: The application uses DEM data to create detailed terrain profiles, crucial for accurate beam blockage calculations.
- **User-Friendly Interface**: Built with Django, the application features an intuitive web interface, making it accessible to users of varying technical expertise.
- **Visualization Tools**: Provides graphical representations of beam blockage data, enabling users to easily interpret and analyze the results.
- **Customizable Parameters**: Users can adjust radar parameters such as range and beam height to tailor simulations to specific needs.
- **Automated PDF Reports**: Utilizing `django-crispy-forms`, `pillow`, and `xhtml2pdf`, the application can generate comprehensive PDF reports of simulation results, including detailed visual and numerical data.
- **Enhanced Form Handling**: `django-crispy-forms` is used to create dynamic and responsive forms, enhancing user interaction and data input.

### Technical Details

#### Technologies Used

- **Django**: A high-level Python web framework for rapid development and clean, pragmatic design.
- **wradlib**: An open-source library for weather radar data processing, providing advanced tools for beam blockage calculations.
- **Python**: The primary programming language for both Django and wradlib, known for its simplicity and versatility.
- **django-crispy-forms**: Used for creating dynamic, responsive, and user-friendly forms.
- **Pillow**: A Python Imaging Library (PIL) fork, used for image processing tasks.
- **xhtml2pdf**: Converts HTML/CSS documents to PDF, facilitating automated report generation.

#### Installation and Setup

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/mhammed2020/djangoradlib.git
   cd djangoradlib
   ```

2. **Set Up the Environment**:
   ```sh
   conda create --name djangoradlib python=3.11
   conda activate djangoradlib
   pip install -r requirements.txt
   ```

3. **Run Migrations**:
   ```sh
   python manage.py migrate
   ```

4. **Start the Development Server**:
   ```sh
   python manage.py runserver
   ```

5. **Access the Application**:
   Open your web browser and navigate to `http://127.0.0.1:8000` to start using DjangoRadlib.

### Example Usage

After setting up the application, users can input the desired radar site coordinates and parameters into the web interface. The application processes this information using the `wradlib` beam blockage function and the provided DEM data to compute fractional beam blockage. Results are displayed interactively, and users can generate detailed PDF reports for documentation and further analysis.

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
    git clone https://github.com/mhammed2020/djangoradlib.git
    cd djangoradlib
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
