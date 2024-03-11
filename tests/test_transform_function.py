import json

def transform_function(original_code, instantiation_dict):
    # Split the original function into lines
    lines = original_code.split('\n')

    # Find the line with the function definition and remove **kwargs
    for i, line in enumerate(lines):
        if line.startswith('def ') and '**kwargs' in line:
            # Replace **kwargs with nothing
            lines[i] = line.replace(', **kwargs', '').replace('**kwargs, ', '').replace('**kwargs', '')

            # Insert the line defining kwargs with the corrected string format
            kwargs_str = json.dumps(instantiation_dict)
            kwargs_line = f"    kwargs = {kwargs_str}"
            lines.insert(i + 1, kwargs_line)
            break

    # Rejoin the modified lines into a single string
    transformed_code = '\n'.join(lines)

    return transformed_code

original_code = """
def py_far(o1, o2, **kwargs):
    center_o1 = ((o1['x1'] + o1['x2']) / 2, (o1['y1'] + o1['y2']) / 2)
    center_o2 = ((o2['x1'] + o2['x2']) / 2, (o2['y1'] + o2['y2']) / 2)
    distance = math.sqrt((center_o1[0] - center_o2[0]) ** 2 + (center_o1[1] - center_o2[1]) ** 2)
    threshold = kwargs.get('threshold', 50)
    return distance > threshold
"""
instantiation_dict = {"threshold": -12.3}
transformed_code = transform_function(original_code, instantiation_dict)
print(transformed_code)

original_code = "def py_behind(o0, o1, **kwargs):\n    # Determine the effective light absorption based on material properties\n    def absorption(material):\n        return kwargs.get('absorption_' + material, 0.5)\n\n    # Calculate the shadow projection of an object on the x-axis from the top-left corner\n    def calculate_shadow(o):\n        return o['x1'] - (o['y1'] * (o['x2'] - o['x1']) / o['y2'])\n    \n    # Check if o0's shadow is within o1's shadow range\n    o0_shadow = calculate_shadow(o0)\n    o1_shadow = calculate_shadow(o1)\n    \n    # o0 is behind o1 if its shadow starts further to the right than o1's shadow ends\n    return o0_shadow >= o1_shadow and absorption(o0['material']) <= absorption(o1['material'])\n"
instantiation_dict = {"absorption_rubber": 0.3, "absorption_metal": 0.7}
transformed_code = transform_function(original_code, instantiation_dict)
print(transformed_code)

