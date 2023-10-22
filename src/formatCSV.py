import numpy as np
import modin.pandas as pd

search_text = '('
replace_text = '_'


def formatCSV(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()
        lines_to_add = []
        for i in range(len(data)):
            if i == 0:
                data[i] = data[i].replace(search_text, replace_text)
                data[i] = data[i].replace(')', '')
                data[i] = data[i].replace('"', '')
                data[i] = data[i].replace('Cell Type', 'CellType')
                data[i] = data[i].replace('CellCenters:0', 'X')
                data[i] = data[i].replace('CellCenters:1', 'Y')
                data[i] = data[i].replace('CellCenters:2', 'Z')
            lines_to_add.append(data[i])

    with open(file_name, 'w') as file:
        file.writelines(lines_to_add)