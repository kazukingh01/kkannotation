from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkannotation*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkannotation',
    version='1.1.0',
    description='my annotation library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukingh01/kkannotation",
    author='kazuking',
    author_email='kazukingh01@gmail.com',
    license='Public License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Private License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy==1.25.2',
        'opencv-python==4.8.0.76',
        'pandas==2.0.3',
        'Pillow==9.5.0',
        'python-dateutil==2.8.2',
        'pytz==2023.3',
        'six==1.16.0',
        'tqdm==4.66.1',
        'joblib==1.3.2',
    ],
    python_requires='>=3.11.2'
)
