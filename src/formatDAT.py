import numpy as np

search_text = '('
replace_text = '_'

c_range = np.arange(0.05, 1, 0.05)


def formatDAT(file_prefix):
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