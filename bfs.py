from collections import deque
import torch

def var_type(t):
    return type(t).__name__

def is_tensor(t):
    return isinstance(t, torch.Tensor)

def shape(t:torch.Tensor):
    return list(t.size())

def bsf_print(y, named_parameters=None, print_saved_tensors=True, print_saved_parameters=True):
    
    named_parameter_pairs = list(named_parameters)
    accounted_address= set()
    parameter_index = dict()

    if named_parameters:
        parameter_index = {tensor:name for name, tensor in named_parameter_pairs}
        accounted_address = { t.untyped_storage().data_ptr() for n,t in named_parameter_pairs}

    stack = deque([y.grad_fn])
    visited = set()
    
    total_cached = 0
    
    print("")
    print("Computation graph nodes:")

    while stack:
        node = stack.popleft()
        if node in visited:
            continue

        visited.add(node)

        if node.name()=='struct torch::autograd::AccumulateGrad':
            tensor_var = node.variable
            assert is_tensor(tensor_var)
            assert tensor_var.requires_grad
            
            # do not add this to activation calculations
            accounted_address.add(tensor_var.untyped_storage().data_ptr())

            if tensor_var in parameter_index:
                print(f"* AccumulateGrad - {parameter_index[tensor_var]} - {list(tensor_var.size())} - dtype: {str(tensor_var.dtype):8} - Addr: {tensor_var.untyped_storage().data_ptr():13}")
            else:
                print(f"* AccumulateGrad - 'Name not known' - {list(tensor_var.size())} - dtype: {str(tensor_var.dtype):8} - Addr: {tensor_var.untyped_storage().data_ptr():13}")
        else:
            print(f"- {node.name()}")
        
        saved_tensor_data = [(atr[7:], getattr(node, atr)) for atr in dir(node) if atr.startswith("_saved_")]
        if saved_tensor_data and (print_saved_tensors or print_saved_parameters):
            var_data = [data for data in saved_tensor_data if not is_tensor(data[1])]
            tensor_data = [data for data in saved_tensor_data if is_tensor(data[1])]

            # handle tensors
            if tensor_data and print_saved_tensors:
                for data in tensor_data:
                    t = data[1]

                    addrr = t.untyped_storage().data_ptr()
                    if addrr not in accounted_address:
                        accounted_address.add(addrr)
                        total_cached += t.untyped_storage().nbytes()
                        # print(f"*** {data[0]:>13} Saved tensor Addr: {addrr} Bytes: {t.untyped_storage().nbytes()}")
                    
                    if t in parameter_index:
                        print(f"[{data[0]:>13}] - dtype: {str(t.dtype):8} - Shape: {str(shape(t)):<15} Tensor: {parameter_index[t]:>15}")
                    else:
                        print(f"[{data[0]:>13}] - dtype: {str(t.dtype)[6:]:8} - Shape: {str(shape(t)):<18} - Addr: {t.untyped_storage().data_ptr():13} - NBytes: {t.untyped_storage().nbytes():>12,} - Size: {t.dtype.itemsize*t.size().numel():>12,}")
            
            # handle variables
            if var_data and print_saved_parameters:
                for data in var_data:
                    print(f"{data[0]:<16} - Type: {str(var_type(data[1])):8} - Value: {str(data[1]):<15}")    

            # print new line if paramters or variables are printed
            if tensor_data and print_saved_tensors  or var_data and print_saved_parameters:
                print("")
                
        stack.extend([next_fn_pair[0] for next_fn_pair in node.next_functions if next_fn_pair[0] is not None and next_fn_pair[0] not in visited])    
    
    print(f"Total cached bytes: {total_cached}")
