export PS1="\u@\h:\w\$ "


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/rds/general/user/lw1824/home/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/rds/general/user/lw1824/home/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/rds/general/user/lw1824/home/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/rds/general/user/lw1824/home/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

eval "$(~/miniforge3/bin/conda shell.bash hook)"

# --- CMD for Requesting Interactive Nodes --- #
interactive() {
    # Set default values
    n=1
    c=4
    m=8
    g=1
    # gpu_type="L40S"
    walltime_hour="4"

    # Process command line arguments
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -n)
                n="$2"
                shift 2
                ;;
            -c)
                c="$2"
                shift 2
                ;;
            -m)
                m="$2"
                shift 2
                ;;
            -g)
                g="$2"
                shift 2
                ;;
            --type)
                gpu_type="$2"
                shift 2
                ;;
            --time)
                walltime_hour="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                return 1
                ;;
        esac
    done

    # Convert the walltime to the proper format (e.g. 08 -> 08:00:00)
    walltime="${walltime_hour}:00:00"

    # Build the command
    cmd="qsub -I -l select=${n}:ncpus=${c}:mem=${m}gb:ngpus=${g} -l walltime=${walltime}"

    # Optionally, print the command before executing it
    echo "Executing: $cmd"

    # Execute the command
    eval "$cmd"
}


# --- CMD for Requesting Interactive Nodes --- #
interactive2a100() {
    # Set default values
    n=1
    c=4
    m=16
    g=2
    gpu_type="A100"
    walltime_hour="12"

    # Process command line arguments
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -n)
                n="$2"
                shift 2
                ;;
            -c)
                c="$2"
                shift 2
                ;;
            -m)
                m="$2"
                shift 2
                ;;
            -g)
                g="$2"
                shift 2
                ;;
            --type)
                gpu_type="$2"
                shift 2
                ;;
            --time)
                walltime_hour="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                return 1
                ;;
        esac
    done

    # Convert the walltime to the proper format (e.g. 08 -> 08:00:00)
    walltime="${walltime_hour}:00:00"

    # Build the command
    cmd="qsub -I -l select=${n}:ncpus=${c}:mem=${m}gb:ngpus=${g}:gpu_type=${gpu_type} -l walltime=${walltime}"

    # Optionally, print the command before executing it
    echo "Executing: $cmd"

    # Execute the command
    eval "$cmd"
}

interactive100() {
    # Set default values
    n=1
    c=4
    m=32
    g=1
    gpu_type="A100"
    walltime_hour="12"

    # Process command line arguments
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -n)
                n="$2"
                shift 2
                ;;
            -c)
                c="$2"
                shift 2
                ;;
            -m)
                m="$2"
                shift 2
                ;;
            -g)
                g="$2"
                shift 2
                ;;
            --type)
                gpu_type="$2"
                shift 2
                ;;
            --time)
                walltime_hour="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                return 1
                ;;
        esac
    done

    # Convert the walltime to the proper format (e.g. 08 -> 08:00:00)
    walltime="${walltime_hour}:00:00"

    # Build the command
    cmd="qsub -I -l select=${n}:ncpus=${c}:mem=${m}gb:ngpus=${g}:gpu_type=${gpu_type} -l walltime=${walltime}"

    # Optionally, print the command before executing it
    echo "Executing: $cmd"

    # Execute the command
    eval "$cmd"
}
