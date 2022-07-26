# Copyright 2021 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


from setuptools import setup, find_packages


setup(
    version='1.0a',
    name='Quagga',
    description='Ot2Rec',
    url='https://github.com/rosalindfranklininstitute/Quagga',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='tests',
    license='Apache License, Version 2.0',
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'mpl_point_clicker',
        'scikit-image',
        'pandas',
        'icecream'
    ],
    # entry_points={
    #     "console_scripts": [
    #     ]
    # }
)
