#
import setuptools
from setuptools import setup


metadata = {'name': 'mayer',
            'maintainer': 'Edward Azizov',
            'maintainer_email': 'edazizovv@gmail.com',
            'description': 'MAYER',
            'license': 'Proprietary',
            'url': 'https://github.com/edazizovv/mayer',
            'download_url': 'https://github.com/edazizovv/mayer',
            'packages': setuptools.find_packages(),
            'include_package_data': True,
            'version': '1.0',
            'long_description': '',
            'python_requires': '>=3.7',
            'install_requires': []}


setup(**metadata)


