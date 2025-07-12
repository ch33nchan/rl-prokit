from setuptools import setup, find_packages

setup(
    name='rl_protokit',
    version='0.1.1',
    packages=find_packages(),  # This will now include 'rl_protokit'
    install_requires=[
        'gymnasium',
        'torch',
        'click',
        'pandas',
        'rich',
    ],
    entry_points={
        'console_scripts': [
            'protokit = rl_protokit.protokit:main',  # Updated path
        ],
    },
    description='A unified RL prototyping toolkit',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/rl-protokit',
)
