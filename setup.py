from setuptools import setup, find_packages

setup(
    name='llm-router',
    version='0.1.1',
    author='herval',
    author_email='hervalfreire@gmail.com',
    description='Quick LLM routing using Embeddings',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
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