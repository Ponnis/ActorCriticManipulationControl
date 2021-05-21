from setuptools import setup

setup(name='inverse_kinematics_gymified',
        version='0.0.1',
        install_requires=['gym'],
        packages=['inverse_kinematics_gymified'],                      # root folder of your package
        package_data={'mypkg': ['envs/arms/*']}  # directory which contains your csvs
)
