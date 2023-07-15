import numpy as np
import modin.pandas as pd

search_text = '('
replace_text = '_'


def formatDAT(file_prefix, c_range):
    for c in c_range:
        file_name = rf'{file_prefix}{"{:.2f}".format(c)}.dat'
        print(f'Editing {file_name} ...')
        with open(file_name, 'r') as file:
            data = file.readlines()
            lines_to_add = []
            for i in range(len(data)):
                if 'ZONE' in data[i]:
                    continue
                data[i] = data[i].replace(search_text, replace_text)
                data[i] = data[i].replace(')', '')
                data[i] = data[i].replace('VARIABLES = ', '')
                lines_to_add.append(data[i])

        with open(file_name, 'w') as file:
            file.writelines(lines_to_add)
        print('done')