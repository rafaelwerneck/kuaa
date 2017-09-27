# Kuaa

Kuaa is a workflow-based framework that can be used for designing, deploying, and executing machine learning experiments. This framework provides a standardized environment for exploratory analysis of machine learning solutions, as it supports the evaluation of feature descriptors, normalizers, classifiers, and fusion approaches in a wide range of tasks involving machine learning.

### Paper

Rafael de Oliveira Werneck, Waldir Rodrigues de Almeida, Bernardo Vecchia Stein, Daniel Vatanabe Pazinato, Pedro Ribeiro Mendes Júnior, Otávio Augusto Bizetto Penatti, Anderson Rocha, Ricardo da Silva Torres, Kuaa: A unified framework for design, deployment, execution, and recommendation of machine learning experiments, In Future Generation Computer Systems, Volume 78, Part 1, 2018, Pages 59-76, ISSN 0167-739X, https://doi.org/10.1016/j.future.2017.06.013. [Bibtex](http://www.recod.ic.unicamp.br/~rwerneck/bibtex/werneck2018kuaa.bib)

--------------

Dependencies
------------


Kuaa dependencies are listed in Kuaa_install_dependencies_v1.3.1.sh.
```
bash Kuaa_install_dependencies_v1.3.1.sh.
```

Interface
---------

To execute Kuaa using its interface:
```
python initInterface.py
```

Terminal
--------
Kuaa can be also executed from the terminal, just need the path to the XML experiment file.
```
python initFramework.py <xml_experiment_path>
```
