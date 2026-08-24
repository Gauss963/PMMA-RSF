import os
import re
import shutil

def delete_pycache() -> None:
    current_directory = os.getcwd()
    pycache_directory = os.path.join(current_directory, '__pycache__')

    if os.path.exists(pycache_directory):
        shutil.rmtree(pycache_directory)

    return None

def read_materials(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    material_blocks = re.findall(r'material\s+(\w+)\s*\[\s*(.*?)\s*\]', content, re.DOTALL)

    materials = {}
    for mat_type, block in material_blocks:
        lines = block.strip().splitlines()
        params = {}
        for line in lines:
            if '=' in line:
                key, val = map(str.strip, line.split('=', 1))
                try:
                    val = eval(val)
                except:
                    pass
                params[key] = val
        name = params.pop('name')
        materials[name] = {
            'type': mat_type,
            'parameters': params
        }

    return materials