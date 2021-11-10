from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkannotation*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkannotation',
    version='1.0.3',
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
        'numpy>=1.20.3',
        'opencv-python>=4.5.3.56',
        'pandas>=1.2.4',
        'python-dateutil>=2.8.1',
        'pytz>=2021.1',
        'six>=1.16.0',
    ],
    python_requires='>=3.7'
)
