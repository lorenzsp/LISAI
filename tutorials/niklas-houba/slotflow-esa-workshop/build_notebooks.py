"""Generate the four workshop notebooks:

    slotflow_tutorial_1_toy.ipynb / slotflow_tutorial_1_solution.ipynb
    slotflow_tutorial_2_diagnose.ipynb / slotflow_tutorial_2_solution.ipynb

    python workshop/build_notebooks.py

Cell content lives in nb_t1.py and nb_t2.py (edit those, not the .ipynb
files). The tutorial variants contain TODO blocks and hints; the solution
variants contain reference implementations and are executed end-to-end
before distribution:

    jupyter nbconvert --to notebook --execute --inplace \\
        workshop/slotflow_tutorial_1_solution.ipynb workshop/slotflow_tutorial_2_solution.ipynb
"""

import os

import nbformat as nbf

import nb_t1
import nb_t2

HERE = os.path.dirname(os.path.abspath(__file__))


def build(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3",
                                 "display_name": "Python 3",
                                 "language": "python"}
    nb.cells = cells
    return nb


if __name__ == "__main__":
    for name, nb in [
        ("slotflow_tutorial_1_toy.ipynb", build(nb_t1.cells(solution=False))),
        ("slotflow_tutorial_1_solution.ipynb", build(nb_t1.cells(solution=True))),
        ("slotflow_tutorial_2_diagnose.ipynb", build(nb_t2.cells(solution=False))),
        ("slotflow_tutorial_2_solution.ipynb", build(nb_t2.cells(solution=True))),
    ]:
        path = os.path.join(HERE, name)
        nbf.write(nb, path)
        print("wrote", path)
