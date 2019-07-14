#!/bin/bash

CBA_FLAGS=
CBA=compss_build_app

usage(){
    show_opts
    exit 0
}

show_opts() {
    cat <<EOT
        Options:

            -h                  Show help options.

            --help              Show help options.

            --ompss             Compile with OmpSs.

            --with_ompss        Set the path to the OmpSs version
                                the user wants to use.

            --ompss-2           Compile with OmpSs-2.

            --with_ompss-2      Set the path to the OmpSs-2 version
                                the user wants to use.

            --cuda              Compile with Cuda.

            --with_cuda         Set the path to the Cuda version
                                the user wants to use.
EOT
}

display_error() {
  local error_msg=$1
  local exitCode=$2

  echo "$error_msg"
  echo " "
 
  exit $exitCode
  #usage $exitCode 
}

# Displays runtime/application errors
error_msg() {
  local error_msg=$1

  # Display error
  echo
  echo "$error_msg"
  echo

  # Exit
  exit 1
}

get_args() {
  # Parse COMPSs Options
  while getopts hvgtmd-: flag; do
    # Treat the argument
    case "$flag" in
      h)
        # Display help
        usage 0
        ;;
      -)
        # Check more complex arguments
        case "$OPTARG" in
          help)
            # Display help
            usage 0
            ;;
          ompss)
            OMPSS=1
          ;;
          ompss-2)
            OMPSS2=1
          ;;
          cuda)
            CUDA=1
          ;;
          with_ompss=*)
            OMPSS=1
            ompss_prefix=${OPTARG//with_ompss=/}
          ;;
          with_ompss-2=*)
            OMPSS2=1
            ompss2_prefix=${OPTARG//with_ompss-2=/}
          ;;
          with_cuda=*)
            CUDA=1
            cuda_prefix=${OPTARG//with_cuda=/}
          ;;
          *)
            # Flag didn't match any patern. Raise exception
            display_error "Bad argument: $OPTARG" 3 #The 3 exit code corresponds with Bad argument
            ;;
        esac
        ;;
      *)
        # Flag didn't match any patern. End of COMPSs flags
        break
        ;;
    esac
  done
  # Shift COMPSs arguments
  shift $((OPTIND-1))

  # Parse application name
  if [[ $# -eq 0 ]]; then
    display_error "Error application name not specified" 3
  else
    other_args=$*
  fi
}

check_ompss_args() {

    cp kmeans-CPU.idl kmeans.idl

    if [ -n "$OMPSS" ] && [ -n "$OMPSS2" ]; then
        echo "[ ERROR ] Impossible to use OmpSs and OmpSs-2 at the same time."
        exit 1
    else
        if [ -n "$CUDA" ]; then
            CBA_FLAGS="$CBA_FLAGS $cuda_prefix --cuda"
        fi
    fi

    if [ -z "$OMPSS" ] && [ -z "$OMPSS2" ]; then
        if [ -n "$CUDA" ]; then
            echo "[ INFO ] CUDA can't be used without using OmpSs or OmpSs-2"
        fi
    else
        CBA_FLAGS="$CBA_FLAGS $ompss_prefix $ompss2_prefix"
    fi

    if [ -n "$OMPSS" ]; then
        CBA_FLAGS="$CBA_FLAGS --ompss"        

        if [ -n "$CUDA" ]; then
            cp src/makefile-ompss-cuda src/makefile
            cp kmeans-CPU-GPU.idl kmeans.idl
        else
            cp src/makefile-ompss src/makefile
        fi
    fi

    if [ -n "$OMPSS2" ]; then
        CBA_FLAGS="$CBA_FLAGS --ompss-2"

        if [ -n "$CUDA" ]; then
            cp kmeans-CPU-GPU.idl kmeans.idl
            cp src/makefile-ompss-2-cuda src/makefile
        else
            cp src/makefile-ompss-2 src/makefile
        fi
    fi

}

check_args() {
    check_ompss_args    
}

get_args "$@"
check_args

#Compilation
echo "-------------- Flags -------------"
echo "CBA_FLAGS   = $CBA_FLAGS"
echo "Application = $other_args"
echo "----------------------------------"

$CBA $CBA_FLAGS $other_args
