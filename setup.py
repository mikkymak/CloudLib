from setuptools import setup, find_packages

setup(
    name='Lbiba',
    packages=find_packages(include=['Lbiba']),
    url='https://github.com/mikkymak/CloudLib.git',
    description='This is a description for Lbiba',
    author='Me',
    license='MIT',
    install_requires=[
        "requests==2.7.0"
        ],
    dependency_links = [
     "git+git://github.com/mikkymak/CloudLib.git#egg=mylib",
    ],
    include_package_data=True,
)
