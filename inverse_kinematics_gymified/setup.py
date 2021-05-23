from setuptools import setup, find_packages

setup(name='inverse_kinematics_gymified',
        version='0.0.1',
        install_requires=['gym'],
#        packages=['inverse_kinematics_gymified'],   
#        packages=[package for package in find_packages()
#            if package.startswith('inverse_kinematics_gymified')],
        packages=find_packages(),

        package_data={'inverse_kinematics_gymified.envs': ['arms/*',
                                                    'data/*']}
        )

