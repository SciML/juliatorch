from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='juliatorch',
      version='0.1.0',
      description='Convert Julia functions to PyTorch autograd functions',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      url='http://github.com/LilithHafner/juliatorch',
      keywords='pytorch julia autograd ad',
      author='Lilith Hafner',
      # author_email='contact@juliadiffeq.org',
      license='MIT',
      packages=['juliatorch','juliatorch.tests'],
      install_requires=['juliacall>=0.9.14', 'torch>=2.1.0', 'numpy>=1.26.1'],
      include_package_data=True,
      zip_safe=False)
