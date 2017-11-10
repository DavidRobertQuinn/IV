from setuptools import setup, find_packages
import os
here = os.path.abspath(os.path.dirname(__file__))

setup(
    name="IV",
    version="0.1",
    packages=find_packages(),
    scripts=['IV/data_extractor.py', 'IV/lightIV.py'],
    author="David Quinn",
    author_email="davidrobertquinn@gmail.com",
    description="Laser Power Transfer Analysis",
    license="Apache License 2.0",
    # cmdclass={'test': PyTest},
    tests_require=['pytest']

)
