.. -*- mode: rst -*-

|Python35|

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg

DBPS
============

DBPS is a Python app for machine learning built on top of
`scikit-learn <http://scikit-learn.org>`_ (The ML system) and `django <https://docs.djangoproject.com>`_ (The UI prediction system) distributed under the 3-Clause BSD license.

The project was started in 2018 by Mordechai Ben Zecharia, As part of collage final project.

Installation
------------

Dependencies
~~~~~~~~~~~~

DBPS requires:

- Python (>= 3.4 the 64bit version)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)
- Matplotlib (>= 1.3.1)
- pandas (>= 0.13.1)
- django (>= 2.0) - for UI prediction system

User installation
~~~~~~~~~~~~~~~~~

Right now there is no official package, you need to clone/download the package in reason to use it (In "Source code" section)
 

Changelog
---------
TODO

Development
-----------

We welcome new contributors of all experience levels. The DBPS
community goals are to be helpful, welcoming, and effective.

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/motybz/Data-based_prediction_system.git


Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Help and Support
----------------

Documentation
~~~~~~~~~~~~~
Steps:

1. Install the dependencies (In the "Dependencies" paragraph).

2. Clone/Download the repository (In the "Source code" paragraph).
3. You need to set the Config file (config.yml).
4. execute the model_investigation.py .
5. edit the MODELS_LOCATION = 'C:/Users/motibz/Documents/Studing/' in ./cancer_prediction/predict_to_user/predict_proba.py according to the output folder (from the config.yml).
6. raise the Django serever (could be run loacly on your machine) - /cancer_prediction/manage.py -runserver ip:port (as appear here: https://docs.djangoproject.com/en/2.0/intro/tutorial01/#the-development-server).
7. go to http://ip:port/user_form/ .
8. Have fun!

Communication
~~~~~~~~~~~~~

TODO
