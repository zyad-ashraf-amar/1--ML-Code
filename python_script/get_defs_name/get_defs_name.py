import ast

def get_function_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()

    tree = ast.parse(file_contents)
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return tuple(function_names)

# Example usage:
fille_path = r'F:\programming languages\My Courses\مبادرة رواد مصر الرقميه\Technical ML\ML Code\4- preprocessing\ML_preprocessing.py'
function_names = get_function_names(fille_path)
formatted_output = ", ".join(function_names)
formatted_output = formatted_output.replace(' ', '\n')
print(formatted_output)

