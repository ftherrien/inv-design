from setuptools import setup

setup(name='didgen',
      version='0.0.0',
      description='Generate molecules with requested properties',
      url='https://github.com/ftherrien/inv-design',
      author='Felix Therrien',
      author_email='felix.therrien@gmail.com',
      license='MIT',
      packages=['didgen'],
      python_requires='>=3',
      install_requires=[
          'numpy',
          'torch',
          'tqdm',
          'matplotlib',
          'rdkit',
          'pyyaml',
      ],
      scripts=['didgenerate'],)
