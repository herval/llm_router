from setuptools import setup, find_packages

setup(
    name='llm-router',
    version='0.1.0',
    author='herval',
    author_email='hervalfreire@gmail.com',
    description='Quick LLM routing using Embeddings',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'chromadb',
    ]
)