from distutils.core import setup
from setuptools import find_packages
with open("README.md", "r") as f:
    long_description = f.read()
setup(name='Build-A-Neural-Network-test',
    version='1.0.0',
    description='A small example package',
    long_description=long_description,
    author='zhaoshidong',
    author_email='18846067738@163.com',
    install_requires=["numpy"],
    license='BSD License',
    packages=find_packages(),
    platforms=["all"],
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries'
    ],
)