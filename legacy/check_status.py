import subprocess
import re
import argparse
from colorama import Fore, Style, init

def get_all_node_names():
    try:
        # Run the sinfo command to get all node names
        command = "sinfo -N -h -o '%N'"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split("\n")
        return output
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_node_state(node_name):
    try:
        # Run the sinfo command to get the node state
        command = f"sinfo -N -h -n {node_name} -o '%T'"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        return output
    except Exception as e:
        print(f"Error: {e}")
        return 'Unknown'

def get_node_resources(node_name):
    try:
        # Run the scontrol command to get node details
        command = f"scontrol show node {node_name}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout

        # Parse CfgTRES and AllocTRES
        cfg_tres_match = re.search(r'CfgTRES=(\S+)', output)
        alloc_tres_match = re.search(r'AllocTRES=(\S+)', output)
        partition_match = re.search(r'Partitions=(\S+)', output)

        if not cfg_tres_match or not partition_match:
            raise ValueError(f"Could not find CfgTRES or Partitions for node {node_name}")

        if not alloc_tres_match:
            print(f"Permission denied for node {node_name}")
            return node_name, None, None, None, 'Unknown', None, 'Unknown', None, None, None, None

        cfg_tres = cfg_tres_match.group(1)
        alloc_tres = alloc_tres_match.group(1)
        partition = partition_match.group(1)

        # Convert CfgTRES and AllocTRES to dictionaries
        cfg_dict = dict(item.split('=') for item in cfg_tres.split(','))
        alloc_dict = dict(item.split('=') for item in alloc_tres.split(','))

        # Parse GPU type
        gpu_type_match = re.search(r'AvailableFeatures=.*GPU_SKU:(\S+)', output)
        gpu_type = gpu_type_match.group(1) if gpu_type_match else 'Unknown'

        # Compute free resources
        total_cpus = int(cfg_dict.get('cpu', 0))
        total_mem = int(cfg_dict.get('mem', '0M')[:-1])
        total_gpus = int(cfg_dict.get('gres/gpu', 0))

        alloc_cpus = int(alloc_dict.get('cpu', 0))
        alloc_mem = int(alloc_dict.get('mem', '0G')[:-1]) * 1024  # Convert GB to MB
        alloc_gpus = int(alloc_dict.get('gres/gpu', 0))

        free_cpus = total_cpus - alloc_cpus
        free_mem = total_mem - alloc_mem
        free_gpus = total_gpus - alloc_gpus

        # Compute utilization percentages, avoid division by zero
        cpu_util = (alloc_cpus / total_cpus) * 100 if total_cpus > 0 else 0
        mem_util = (alloc_mem / total_mem) * 100 if total_mem > 0 else 0
        gpu_util = (alloc_gpus / total_gpus) * 100 if total_gpus > 0 else 0

        # Get node state
        node_state = get_node_state(node_name)

        return node_name, partition, free_cpus, free_mem / 1024, free_gpus, gpu_type, cpu_util, mem_util, gpu_util, total_gpus, node_state
    except ValueError as ve:
        print(ve)
        return node_name, None, None, None, 'Unknown', None, 'Unknown', None, None, None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return node_name, None, None, None, 'Unknown', None, 'Unknown', None, None, None, None, None

def parse_gpu_info(gpu_info):
    if gpu_info == 'Unknown':
        return 'N/A', 'N/A', 'N/A'
    parts = gpu_info.split(',')
    gpu_type = parts[0].replace('_', ' ')
    gpu_mem = next((p.split(':')[1] for p in parts if 'GPU_MEM' in p), 'N/A')
    gpu_cc = next((p.split(':')[1] for p in parts if 'GPU_CC' in p), 'N/A')
    return gpu_type, gpu_mem, gpu_cc

def print_table(node_resources):
    header = f"{'Node Name':<20} {'Partition':<15} {'Free CPUs':<10} {'Free Memory (GB)':<20} {'Free GPUs':<10} {'GPU Type':<15} {'GPU Memory':<10} {'GPU CC':<10} {'State':<10}"
    print(header)
    print("=" * len(header))
    
    for node_name, partition, free_cpus, free_mem, free_gpus, gpu_type, cpu_util, mem_util, gpu_util, total_gpus, node_state in node_resources:
        if free_cpus is not None and free_mem is not None:
            # Determine color based on utilization and state
            if node_state.lower() == 'down':
                node_name_color = Fore.RED
            else:
                node_name_color = Fore.RESET

            if cpu_util == 0 and mem_util == 0 and gpu_util == 0:
                color = Fore.GREEN
            else:
                color = Fore.RESET
                cpu_color = Fore.RED if cpu_util > 75 else Fore.GREEN if cpu_util == 0 else Fore.YELLOW
                mem_color = Fore.RED if mem_util > 75 else Fore.GREEN if mem_util == 0 else Fore.YELLOW
                gpu_color = Fore.RED if gpu_util > 75 or free_gpus == 0 else Fore.GREEN if gpu_util == 0 else Fore.YELLOW
            
            # Parse GPU info
            gpu_type_str, gpu_mem_str, gpu_cc_str = parse_gpu_info(gpu_type)
            
            # Check if the entire row is "N/A"
            if free_cpus == 0 and free_mem == 0 and free_gpus == 0 and gpu_type_str == 'N/A':
                continue
            
            free_gpus_display = 'N/A' if total_gpus == 0 else free_gpus

            print(f"{node_name_color}{node_name:<20}{Style.RESET_ALL} "
                  f"{partition:<15} "
                  f"{color if color == Fore.GREEN else cpu_color}{free_cpus:<10}{Style.RESET_ALL} "
                  f"{color if color == Fore.GREEN else mem_color}{free_mem:<20.2f}{Style.RESET_ALL} "
                  f"{Fore.WHITE if free_gpus_display == 'N/A' else Fore.RED if free_gpus_display == 0 else gpu_color}{free_gpus_display:<10}{Style.RESET_ALL} "
                  f"{Fore.WHITE if gpu_type_str == 'N/A' else Fore.RESET}{gpu_type_str:<15}{Style.RESET_ALL} "
                  f"{Fore.WHITE if gpu_mem_str == 'N/A' else Fore.RESET}{gpu_mem_str:<10}{Style.RESET_ALL} "
                  f"{Fore.WHITE if gpu_cc_str == 'N/A' else Fore.RESET}{gpu_cc_str:<10}{Style.RESET_ALL} "
                  f"{node_state:<10}")
        else:
            continue

def main():
    init(autoreset=True)  # Initialize colorama
    parser = argparse.ArgumentParser(description="List available resources on specified Slurm nodes.")
    parser.add_argument("nodes", metavar="N", type=str, nargs="*", help="List of node names")
    parser.add_argument("--all", action="store_true", help="Print all information about the Slurm cluster")
    args = parser.parse_args()

    if args.all:
        nodes = get_all_node_names()
    else:
        nodes = args.nodes

    node_resources = []
    for node in nodes:
        node_resources.append(get_node_resources(node))
    
    print_table(node_resources)

if __name__ == "__main__":
    main()