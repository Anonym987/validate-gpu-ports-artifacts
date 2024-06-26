#!/bin/bash
MSGPACK_VERSION=5.0.0

if [ -z ${EXTRA_PROF_COMPILER+y} ]; then
    echo >&2 "EXTRA PROF ERROR: No compiler set, either set EXTRA_PROF_COMPILER or use one of the compiler-specific wrappers."
    exit 1
fi

if [ "${EXTRA_PROF_WRAPPER}" = off ] || [ "${EXTRA_PROF_WRAPPER}" = OFF ]; then
    exec $EXTRA_PROF_COMPILER "$@"
fi
arguments=("$@")
compile_only=false
shared_library=false
while [ $# -gt 0 ]; do
    #echo "DEBUG: $#: $1"
    case "$1" in
    -c)
        compile_only=true
        ;;
    --compile)
        compile_only=true
        ;;
    -shared)
        shared_library=true
        ;;
    -ccbin)
        compiler_dir="$1 $2"
        ;;
    -ccbin=*)
        compiler_dir=$1
        ;;
    --compiler-bindir=*)
        compiler_dir=$1
        ;;
    esac
    shift
done

extra_prof_root="$(dirname "${BASH_SOURCE[0]}")"
msg_pack_root="$extra_prof_root/msgpack"
instrumentation_arguments=("-g")
extra_prof_arguments=("$extra_prof_optimization" "--std=c++17" "-DMSGPACK_NO_BOOST=1" "-I$msg_pack_root/include" "$extra_prof_event_trace")
link_extra_prof_wrap="-Xlinker --no-as-needed -Xlinker --rpath=\$ORIGIN -l_extra_prof -Xlinker --as-needed -L."

#if $shared_library; then
extra_prof_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fPIC")
#fi

if [ $EXTRA_PROF_COMPILER = "nvcc" ]; then
    extra_prof_arguments+=("$compiler_dir")
fi

if [ "${EXTRA_PROF_GPU}" != "off" ] && [ "${EXTRA_PROF_GPU}" != "OFF" ]; then
    extra_prof_arguments+=("-DEXTRA_PROF_GPU=1")
fi

if [ "${EXTRA_PROF_ENERGY}" = on ] || [ "${EXTRA_PROF_ENERGY}" = ON ]; then
    extra_prof_arguments+=("-DEXTRA_PROF_ENERGY=1")
fi

if [ "${EXTRA_PROF_EVENT_TRACE}" = on ] || [ "${EXTRA_PROF_EVENT_TRACE}" = ON ]; then
    extra_prof_arguments+=("-DEXTRA_PROF_EVENT_TRACE=1")
fi

if [ "${EXTRA_PROF_DEBUG_BUILD}" = on ] || [ "${EXTRA_PROF_DEBUG_BUILD}" = ON ]; then
    extra_prof_arguments+=("-g -O0")
else
    extra_prof_arguments+=("-O3")
fi

if [ "${EXTRA_PROF_DEBUG_SANITIZE}" = on ] || [ "${EXTRA_PROF_DEBUG_SANITIZE}" = ON ]; then
    instrumentation_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fsanitize=address")
    extra_prof_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fsanitize=address")
fi

if [ "${EXTRA_PROF_DEBUG_INSTRUMENTATION}" = on ] || [ "${EXTRA_PROF_DEBUG_INSTRUMENTATION}" = ON ]; then
    extra_prof_arguments+=("-DEXTRA_PROF_DEBUG_INSTRUMENTATION=1")
fi

if [ -z ${EXTRA_PROF_SCOREP_INSTRUMENTATION+y} ]; then
    instrumentation_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-finstrument-functions")

    if [ "${EXTRA_PROF_ADVANCED_INSTRUMENTATION}" != "off" ] && [ "${EXTRA_PROF_ADVANCED_INSTRUMENTATION}" != "OFF" ] && [ -z ${EXTRA_PROF_SCOREP_INSTRUMENTATION+y} ]; then
        #find path to headers that should be excluded
        combined=("$compiler_dir" "$EXTRA_PROF_COMPILER_OPTION_REDIRECT" -H "$EXTRA_PROF_COMPILER_OPTION_REDIRECT" -E $extra_prof_root/extra_prof/compiler_information/find_libstd_include.cpp)
        echo "EXTRA PROF FIND INCLUDES: " $EXTRA_PROF_COMPILER ${combined[*]}
        potential_include_paths=$($EXTRA_PROF_COMPILER ${combined[*]} 2>&1)

        declare -a all_include_paths
        while IFS= read -r line; do
            if [[ $line == .* ]]; then
                line_without_dot="${line#*. }" # Remove the dot and space from the beginning of the line

                if [[ $line_without_dot != *"/tmp/"* ]]; then
                    all_include_paths+=($(dirname "$line_without_dot"))
                fi
            fi
        done <<<"$potential_include_paths"

        path_string=$(printf "%s\n" "${all_include_paths[@]}")
        IFS=$'\n' read -rd '' -a unique_include_paths <<<"$(echo "$path_string" | sort -r -u)"

        remove_entries() {
            local -n arr=$1
            local prev_entry=${arr[-1]}
            for ((i = ${#arr[@]} - 2; i > 0; i--)); do
                if [[ ${arr[$i]} == "$prev_entry"* ]]; then
                    unset 'arr[i]' # Remove the entry if it begins with the previous entry
                else
                    prev_entry=${arr[$i]}
                fi
            done
        }

        remove_entries unique_include_paths

        for unique_include_path in "${unique_include_paths[@]}"; do
            instrumentation_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-finstrument-functions-exclude-file-list=$unique_include_path")
        done

    fi #end EXTRA_PROF_ADVANCED_INSTRUMENTATION

    if [ -z ${EXTRA_PROF_EXCLUDE_FILES+y} ]; then
        :
    else
        instrumentation_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-finstrument-functions-exclude-file-list=$EXTRA_PROF_EXCLUDE_FILES")
    fi

    if [ -z ${EXTRA_PROF_EXCLUDE_FUNCTIONS+y} ]; then
        :
    else
        instrumentation_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-finstrument-functions-exclude-function-list=$EXTRA_PROF_EXCLUDE_FUNCTIONS")
    fi

else
    instrumentation_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fplugin=$EXTRA_PROF_SCOREP_INSTRUMENTATION")
    extra_prof_arguments+=("-DEXTRA_PROF_SCOREP_INSTRUMENTATION=1")
fi

if $compile_only; then
    combined=("${instrumentation_arguments[@]}" "${arguments[@]}")
    echo "EXTRA PROF COMPILE: " $EXTRA_PROF_COMPILER ${combined[*]}
    exec $EXTRA_PROF_COMPILER ${combined[*]}
else
    if [ -d "$extra_prof_root/msgpack" ]; then
        echo "EXTRA PROF: Found msgpack"
    else
        echo "EXTRA PROF: Downloading msgpack..."
        wget -nv -O"$extra_prof_root/msgpack-cxx.tar.gz" "https://github.com/msgpack/msgpack-c/releases/download/cpp-$MSGPACK_VERSION/msgpack-cxx-$MSGPACK_VERSION.tar.gz"
        [ $? -eq 0 ] || exit $?
        echo "EXTRA PROF: Unpacking msgpack..."
        tar -xzf "$extra_prof_root/msgpack-cxx.tar.gz" -C "$extra_prof_root"
        [ $? -eq 0 ] || exit $?
        mv "$extra_prof_root/msgpack-cxx-$MSGPACK_VERSION" "$msg_pack_root"
        [ $? -eq 0 ] || exit $?
        echo "EXTRA PROF: Finished unpacking msgpack. Continuing..."
    fi

    # no exec otherwise this script will end here
    combined=("-c" "${extra_prof_arguments[@]}" "-o" "extra_prof_injection.o" "$extra_prof_root/extra_prof/injection/injection.cpp")
    echo "EXTRA PROF COMPILE INJECTION: " $EXTRA_PROF_COMPILER ${combined[*]}
    $EXTRA_PROF_COMPILER ${combined[*]}
    [ $? -eq 0 ] || exit $?

    # compile library
    combined=("--shared" "${extra_prof_arguments[@]}" "$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fPIC" "$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fopenmp" "-o" "lib_extra_prof.so" "$extra_prof_root/extra_prof/instrumentation/instrumentation.cpp" "$extra_prof_root/extra_prof/library/lib_extra_prof.cpp")

    echo "EXTRA PROF COMPILE LIBRARY: " $EXTRA_PROF_COMPILER ${combined[*]}
    $EXTRA_PROF_COMPILER ${combined[*]}
    [ $? -eq 0 ] || exit $?

    #
    combined=("$link_extra_prof_wrap" "extra_prof_injection.o" "${instrumentation_arguments[@]}" "${arguments[@]}")

    if [ "${EXTRA_PROF_GPU}" != "off" ] && [ "${EXTRA_PROF_GPU}" != "OFF" ]; then

        for cuda_var_name in CUDA_HOME CUDA_ROOT CUDAHOME CUDAROOT CUDA; do
            cuda_var_value="$(eval echo \$$cuda_var_name)"

            if [ -n "$cuda_var_value" ]; then
                CUDA_HOME="$cuda_var_value"
                echo "EXTRA PROF: Found $cuda_var_name: $cuda_var_value"
                break
            fi
        done
        if [ -z "$CUDA_HOME" ]; then
            CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
            echo "EXTRA PROF: Detected CUDA_HOME via nvcc: $CUDA_HOME"
        fi

        combined+=("-lcupti -lcuda -lnvperf_host -lnvperf_target -L$CUDA_HOME/extras/CUPTI/lib64")
        if [ "${EXTRA_PROF_ENERGY}" = on ] || [ "${EXTRA_PROF_ENERGY}" = ON ]; then
            combined+=("-lnvidia-ml")
        fi
    fi

    echo "EXTRA PROF CALL: " $EXTRA_PROF_COMPILER ${combined[*]}
    exec $EXTRA_PROF_COMPILER ${combined[*]}
fi
