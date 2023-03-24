from setuptools import setup

setup(name='quanser_robots',
      version='0.1.1',
      description='Simulators and control interfaces for Quanser platforms',
      url='https://git.ias.informatik.tu-darmstadt.de/quanser/clients',
      author='Intelligent Autonomous Systems Lab',
      author_email='boris@robot-learning.de',
      license='MIT',
      packages=['quanser_robots', 'quanser_robots.qube', 'quanser_robots.cartpole'],
      zip_safe=False,
      install_requires=[
          'gym>=0.10.5', 'numpy>=1.14.5', 'scipy>=1.1.0', 'matplotlib>=2.2.3',
          'vpython>=7.4.6','pynput>=1.4.0'
      ])
