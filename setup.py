from setuptools import setup

setup(
    name='keras-condenser',
    version='0.0.2',
    description='Seq2Vec layer on Tensorflow by summarizing feature distribution with characteristic function.',
    py_modules=['condenser'],
    install_requires=['tensorflow'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ]
)
