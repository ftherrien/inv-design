from setuptools import setup, find_packages

setup(name='didgen',
      version='1.0.1',
      description='Generate molecules with requested properties',
      url='https://github.com/ftherrien/inv-design',
      author='Felix Therrien',
      author_email='felix.therrien@gmail.com',
      license='GNU GPLv3',
      packages=find_packages(),
      python_requires='>=3',
      install_requires=[
          'numpy',
          'torch',
          'tqdm',
          'matplotlib',
          'rdkit',
          'pyyaml',
          'bayesian-optimization',
          'torch_geometric',
      ],
      scripts=['didgenerate'],)
