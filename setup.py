from distutils.core import setup

setup(
    name="pangolin",
    python_requires=">3.11",
    packages=["pangolin", "pangolin.ir", "pangolin.interface", "pangolin.inference"],
)
