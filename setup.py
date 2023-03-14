from setuptools import setup, find_packages

setup(
    name='dabryvisu',
    version='1.0.1',
    description='Visualize trajectory optimization in flow fields',
    author='Bastien Schnitzler',
    author_email='bastien.schnitzler@live.fr',
    packages=find_packages('src'),  # same as name
    package_dir={'': 'src'},
    install_requires=[
        'basemap',
        'easygui',
        'eel',
        'geopy',
        'h5py',
        'ipython',
        'ipywidgets',
        'matplotlib',
        'mpld3',
        'numpy',
        'pyproj',
        'PyQt5',
        'scipy',
        'tqdm'],
)
