import pandas as pd

def skill_retrieve(path):
    employees = pd.read_csv(path)
    position = list(employees['Position'])
    positions = set(','.join(position).split(sep=","))
    return positions