from lexer import Lexer
from parser import Parser
from intermediateRepresentation import SSAGenerator, ConstInstruction, AddInstruction, SubInstruction, MulInstruction, DivInstruction, WriteInstruction, WriteNLInstruction, ReadInstruction, BranchInstruction, JumpInstruction, CompareInstruction, PhiInstruction, CallInstruction, ReturnInstruction
import subprocess
import sys 

def optimize(ssa: SSAGenerator):
    # Remove unnecessary initialization constants first
    for block in ssa.blocks:
        new_instructions = []
        for inst in block.instructions:
            if not (isinstance(inst, ConstInstruction) and 
                   inst.value == 0 and 
                   not any(inst.result == other.left or inst.result == other.right 
                          for other in block.instructions 
                          if hasattr(other, 'left') or hasattr(other, 'right'))):
                new_instructions.append(inst)
        block.instructions = new_instructions
    
    # Run optimizations with iteration limit
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        changed = False
        old_instruction_count = sum(len(block.instructions) for block in ssa.blocks)
        
        # Run optimization passes
        changed |= do_constant_folding(ssa)
        changed |= do_copy_propagation(ssa)
        changed |= do_common_subexpression_elimination(ssa)
        
        # Check if we actually changed anything
        new_instruction_count = sum(len(block.instructions) for block in ssa.blocks)
        if not changed or new_instruction_count >= old_instruction_count:
            break
            
        iteration += 1
        
def do_constant_folding(ssa: SSAGenerator) -> bool:
    changed = False
    # Keep track of known constant values across all blocks
    constant_values = {}
    
    # First pass: collect all constant values
    for block in ssa.blocks:
        for inst in block.instructions:
            if isinstance(inst, ConstInstruction):
                constant_values[inst.result] = inst.value
    
    # Second pass: fold constant expressions
    for block in ssa.blocks:
        new_instructions = []
        for inst in block.instructions:
            if isinstance(inst, (AddInstruction, SubInstruction, MulInstruction, DivInstruction)):
                # Get constant values if they exist
                left_const = constant_values.get(inst.left)
                right_const = constant_values.get(inst.right)
                
                # If both operands are constants, fold the expression
                if left_const is not None and right_const is not None:
                    if isinstance(inst, AddInstruction):
                        value = left_const + right_const
                    elif isinstance(inst, SubInstruction):
                        value = left_const - right_const
                    elif isinstance(inst, MulInstruction):
                        value = left_const * right_const
                    elif isinstance(inst, DivInstruction) and right_const != 0:
                        value = left_const // right_const
                    else:
                        new_instructions.append(inst)
                        continue
                        
                    new_inst = ConstInstruction(value, inst.result)
                    new_instructions.append(new_inst)
                    constant_values[inst.result] = value
                    changed = True
                else:
                    new_instructions.append(inst)
            else:
                new_instructions.append(inst)
                
        block.instructions = new_instructions
    
    return changed

def do_copy_propagation(ssa: SSAGenerator) -> bool:
    changed = False
    copies = {}
    
    # First collect all copies across blocks
    for block in ssa.blocks:
        for inst in block.instructions:
            if isinstance(inst, ConstInstruction):
                copies[inst.result] = inst
    
    # Now propagate copies
    for block in ssa.blocks:
        new_instructions = []
        for inst in block.instructions:
            if isinstance(inst, (AddInstruction, SubInstruction, MulInstruction, DivInstruction, CompareInstruction)):
                # Replace operands with their copies if available
                new_left = inst.left
                new_right = inst.right
                
                if inst.left in copies:
                    new_left = copies[inst.left].result
                    changed = True
                if inst.right in copies:
                    new_right = copies[inst.right].result
                    changed = True
                
                if new_left != inst.left or new_right != inst.right:
                    new_inst = type(inst)(new_left, new_right, inst.result)
                    new_instructions.append(new_inst)
                else:
                    new_instructions.append(inst)
            elif isinstance(inst, WriteInstruction):
                if inst.value in copies:
                    new_instructions.append(WriteInstruction(copies[inst.value].result))
                    changed = True
                else:
                    new_instructions.append(inst)
            else:
                new_instructions.append(inst)
                
        block.instructions = new_instructions
    
    return changed

def do_common_subexpression_elimination(ssa: SSAGenerator) -> bool:
    changed = False
    
    for block in ssa.blocks:
        new_instructions = []
        expressions = {}  # Maps canonical form to original instruction
        value_map = {}   # Maps SSA values to their replacements
        
        for inst in block.instructions:
            if isinstance(inst, WriteInstruction):
                # Find the most recent value in phi nodes
                write_value = inst.value
                latest_value = None
                
                # Look through phi instructions to find latest value
                for phi_inst in reversed(new_instructions):
                    if isinstance(phi_inst, PhiInstruction) and write_value in phi_inst.values:
                        latest_value = phi_inst.result
                        break
                
                if latest_value:
                    new_instructions.append(WriteInstruction(latest_value))
                else:
                    new_instructions.append(inst)
            elif isinstance(inst, (BranchInstruction, JumpInstruction, CompareInstruction,
                               ConstInstruction, ReadInstruction, WriteNLInstruction)):
                new_instructions.append(inst)
    
    # Then eliminate common subexpressions
    for block in ssa.blocks:
        new_instructions = []
        block_expressions = {}  # Track expressions within this block
        
        for inst in block.instructions:
            if isinstance(inst, (BranchInstruction, JumpInstruction, CompareInstruction,
                               ConstInstruction, ReadInstruction, WriteNLInstruction)):
                new_instructions.append(inst)
            elif isinstance(inst, WriteInstruction):
                # Update write instruction value if it has been mapped
                mapped_value = value_map.get(inst.value, inst.value)
                new_instructions.append(WriteInstruction(mapped_value))
            elif isinstance(inst, (MulInstruction, AddInstruction)):
                # Get potentially remapped operands
                left = value_map.get(inst.left, inst.left)
                right = value_map.get(inst.right, inst.right)
                
                # Create canonical form for commutative operations
                operands = tuple(sorted([left, right]))
                key = (type(inst), operands)
                
                if key in block_expressions:
                    # Found a common subexpression in this block
                    value_map[inst.result] = block_expressions[key].result
                    changed = True
                else:
                    # New expression for this block
                    new_inst = type(inst)(left, right, inst.result)
                    block_expressions[key] = new_inst
                    new_instructions.append(new_inst)
                    
                    # Check if it exists in another block
                    if key in expressions and expressions[key].result != inst.result:
                        value_map[inst.result] = expressions[key].result
                        changed = True
            else:
                new_instructions.append(inst)
                
        block.instructions = new_instructions
    
    return changed

def generate_dot(ssa: SSAGenerator) -> str:
    dot = [
        "digraph G {",
        "  node [shape=record];",
        "  rankdir=TB;",
        "  nodesep=0.7;",
        "  ranksep=0.7;",
        
        # Create subgraphs for each function
        "  compound=true;",  # Enable connections between subgraphs
    ]
    
    # Track node names to avoid conflicts
    node_names = {}
    
    # Generate subgraphs for each function
    for func_name, blocks in ssa.functions.items():
        if not blocks:  # Skip empty functions
            continue
            
        # Create subgraph for this function
        dot.append(f'  subgraph cluster_{func_name} {{')
        dot.append(f'    label = "Function: {func_name}";')
        dot.append('    style = rounded;')  # Optional: gives rounded corners
        dot.append('    color = blue;')     # Optional: colored border
        
        # Add nodes for this function
        for block in blocks:
            instructions = []
            for inst in block.instructions:
                inst_str = str(inst).replace('"', '\\"').replace('<', '\\<').replace('>', '\\>')
                instructions.append(inst_str)
                
            if not instructions:
                instructions = ["\\<empty\\>"]
            
            instr_label = " | ".join(instructions)
            node_name = f"{func_name}_bb{block.id}"
            node_names[block] = node_name
            label = f'"BB{block.id} | {{{instr_label}}}"'
            dot.append(f'    {node_name} [label={label}];')
        
        # Add edges between blocks in this function
        for block in blocks:
            last_inst = block.get_last_instruction()
            if isinstance(last_inst, BranchInstruction):
                dot.append(f'    {node_names[block]} -> {func_name}_bb{last_inst.true_target} [label="true"];')
                dot.append(f'    {node_names[block]} -> {func_name}_bb{last_inst.false_target} [label="false"];')
            elif isinstance(last_inst, JumpInstruction):
                dot.append(f'    {node_names[block]} -> {func_name}_bb{last_inst.target} [label=""];')
            elif isinstance(last_inst, ReturnInstruction):
                # Optional: highlight return points
                dot.append(f'    {node_names[block]} [color=green];')
            
        dot.append('  }')  # End subgraph
    
    # Add edges for function calls between subgraphs
    for func_name, blocks in ssa.functions.items():
        for block in blocks:
            for inst in block.instructions:
                if isinstance(inst, CallInstruction):
                    # Add edge from caller to called function's entry block
                    called_func = inst.function
                    if called_func in ssa.functions and ssa.functions[called_func]:
                        called_entry = f"{called_func}_bb0"
                        dot.append(f'  {node_names[block]} -> {called_entry} [label="call", style=dashed, color=red];')
    
    dot.append("}")
    return "\n".join(dot)

def print_ssa_form(ssa: SSAGenerator):
    """Print SSA form with clean formatting, separated by functions"""
    print("SSA form:")
    for func_name, blocks in ssa.functions.items():
        if not blocks:  # Skip empty functions
            continue
        print(f"\nFunction: {func_name}")
        for block in blocks:
            print(f"BB{block.id}:")
            for inst in block.instructions:
                comment = ""
                if isinstance(inst, (AddInstruction, SubInstruction, MulInstruction, DivInstruction)):
                    comment = f"# {inst.left} {inst.__class__.__name__[0]} {inst.right}"
                print(f"  {str(inst)} {comment}")

def compile_and_run(text: str):
    try:
        # Lexical and Syntax Analysis
        lexer = Lexer(text)
        parser = Parser(lexer)
        ast = parser.program()
        
        # Generate SSA IR
        ssa_gen = SSAGenerator()
        ssa_gen.visit(ast)
        
        # Print initial SSA form
        print("\nInitial SSA form:")
        print_ssa_form(ssa_gen)
        
        # Optimize
        optimize(ssa_gen)
        
        # Print optimized form
        print("\nOptimized SSA form:")
        print_ssa_form(ssa_gen)
        
        # Generate and save visualization
        generate_and_save_dot(ssa_gen)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
        # Comment out the execution part:
        # print("\nProgram Started:")
        # interpreter = IRInterpreter(ssa_gen.blocks)
        # interpreter.run()
        
    except Exception as e:
        print(f"Error: {str(e)}")

def generate_and_save_dot(ssa: SSAGenerator, output_file: str = "output"):
    """Generate DOT graph, print it, and save as PNG"""
    # Generate the DOT content
    dot = [
        "digraph G {",
        "  node [shape=record];",
        "  rankdir=TB;",
        "  nodesep=0.7;",
        "  ranksep=0.7;",
        "  compound=true;"  # Enable connections between subgraphs
    ]
    
    # Create subgraphs for each function
    for func_name, blocks in ssa.functions.items():
        if not blocks:  # Skip empty functions
            continue
            
        # Create subgraph for this function
        dot.append(f'  subgraph cluster_{func_name} {{')
        dot.append(f'    label = "Function: {func_name}";')
        dot.append('    style = rounded;')
        dot.append('    color = blue;')
        
        # Add nodes for this function
        for block in blocks:
            instructions = []
            for inst in block.instructions:
                inst_str = str(inst).replace('"', '\\"').replace('<', '\\<').replace('>', '\\>')
                instructions.append(inst_str)
                
            if not instructions:
                instructions = ["\\<empty\\>"]
            
            instr_label = " | ".join(instructions)
            label = f'"{func_name}_bb{block.id}" [label="BB{block.id} | {{{instr_label}}}"];'
            dot.append(f'    {label}')
            
            # Add edges between blocks
            last_inst = block.get_last_instruction()
            if isinstance(last_inst, BranchInstruction):
                dot.append(f'    {func_name}_bb{block.id} -> {func_name}_bb{last_inst.true_target} [label="true"];')
                dot.append(f'    {func_name}_bb{block.id} -> {func_name}_bb{last_inst.false_target} [label="false"];')
            elif isinstance(last_inst, JumpInstruction):
                dot.append(f'    {func_name}_bb{block.id} -> {func_name}_bb{last_inst.target} [label=""];')
        
        dot.append('  }')
    
    # Add edges for function calls
    for func_name, blocks in ssa.functions.items():
        for block in blocks:
            for inst in block.instructions:
                if isinstance(inst, CallInstruction) and inst.function in ssa.functions:
                    # Add edge from caller to called function
                    dot.append(f'  {func_name}_bb{block.id} -> {inst.function}_bb0 [label="call", style=dashed, color=red];')
    
    dot.append("}")
    dot_content = "\n".join(dot)
    
    # Print the DOT content
    print("\nGenerated DOT graph:")
    print(dot_content)
    
    try:
        # Save as PNG
        import subprocess
        dot_process = subprocess.Popen(['dot', '-Tpng'], 
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        
        # Send dot content and get PNG output
        png_output, errors = dot_process.communicate(input=dot_content.encode())
        
        if dot_process.returncode != 0:
            print(f"Error running dot: {errors.decode()}")
            return
            
        # Save PNG to file
        with open(f"{output_file}.png", 'wb') as f:
            f.write(png_output)
            
        print(f"\nSaved PNG visualization to {output_file}.png")
        
    except FileNotFoundError:
        print("Error: 'dot' command not found. Please make sure Graphviz is installed.")
    except Exception as e:
        print(f"Error saving PNG: {str(e)}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python compiler.py <input_file.tiny>")
        return
        
    input_file = sys.argv[1]
    if not input_file.endswith('.tiny'):
        print("Error: Input file must have .tiny extension")
        return
        
    try:
        with open(input_file, 'r') as f:
            program = f.read()
        compile_and_run(program)
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except Exception as e:
        print(f"Error during compilation: {str(e)}")

if __name__ == "__main__":
    main()
