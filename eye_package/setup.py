import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'eye_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

    # ---------------- 添加 launch 文件支持 ----------------
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team-3',
    maintainer_email='alyml50@nottingham.ac.uk',
    description='Navigation package for robot waypoint following',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'eye_node = eye_package.eye_node:main',
            'nav_node = eye_package.nav_node:main',
        ],
    },
)

