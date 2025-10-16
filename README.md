# Project 2 FYS-STK4155
### Fall 2025
**Authors:** *Ingvild Olden Bjerkelund, Kjersti Stangeland, Jenny Guldvog, & Sverre Manu Johansen*

**Using functions:** 

If using .py-files:
    To use the functions package in your own folder, paste:

    from functions import *

    in your file, then run your code in terminal from the code from root "../Code/":

    python -m Sverre.test *or* python -m your_folder.your_file

Elif using .ipynb-files:
    Paste this into your notebook with your other packages:
        import sys, os

        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        sys.path.append(project_root)

        from functions import *