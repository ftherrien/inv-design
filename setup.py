from setuptools import setup, find_packages

setup(name='didgen',
      version='0.0.0',
      description='Generate molecules with requested properties',
      url='https://github.com/ftherrien/inv-design',
      author='Felix Therrien',
      author_email='felix.therrien@gmail.com',
      license='MIT',
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
