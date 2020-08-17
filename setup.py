from distutils.core import setup
setup(
  name = 'flexgan',
  packages = ['flexgan'],
  version = '0.1',
  license='MIT',
  description = 'The FlexGAN library provides autonomous synthetic data generation for structured data sets.',
  author = 'Eric Muccino',
  author_email = 'emuccino@mindboard.com',
  url = 'https://github.com/emuccino/flexgan',
  download_url = 'https://github.com/emuccino/flexgan/archive/0.1.tar.gz',
  keywords = ['SYNTHETIC', 'DATA', 'GENERATION','GENERATE','GENERATIVE'],
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'tensorflow',
          'scikit-learn',
          'scipy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)