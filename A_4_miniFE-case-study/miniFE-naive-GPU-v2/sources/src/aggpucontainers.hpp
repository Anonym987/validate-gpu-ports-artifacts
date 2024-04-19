#pragma once

#include <set>
#include <map>
#include <agcudacomon.hpp>

template <typename T>
struct gpu_set
{
    T *data;
    size_t size;

    gpu_set(const std::set<T> &org_set)
    {
        size = org_set.size();
        CHECK_CUDA_ERROR(cudaMallocManaged(&data, size * sizeof(T)));
        int counter = 0;
        for (auto &&val : org_set)
        {
            data[counter] = val;
            counter++;
        }
        cudaMemPrefetchAsync(data, size * sizeof(T), 0);
    }

    ~gpu_set()
    {
#ifndef __CUDACC__
        cudaFree(data);
        data = nullptr;
        size = 0;
#endif
    }

    __device__ T *lower_bound(const T &value) const
    {
        T *it;
        ptrdiff_t count, step;
        count = size;
        auto first = data;

        while (count > 0)
        {
            it = first;
            step = count / 2;
            it += step;

            if (*it < value)
            {
                first = ++it;
                count -= step + 1;
            }
            else
                count = step;
        }

        return first;
    }

    __device__ int find(const T &x) const
    {
        T *res = lower_bound(x);
        if (res == data + size)
        {
            return end();
        }
        else if (*res == x)
        {
            return res - data;
        }
        return end();
    }

    __device__ int end() const
    {
        return -1;
    }
};

template <typename K, typename V>
struct gpu_map
{
    struct pair
    {
        K first;
        V second;
    };

    pair *data;
    size_t size;

    gpu_map(const std::map<K, V> &org_map)
    {
        size = org_map.size();
        CHECK_CUDA_ERROR(cudaMallocManaged(&data, size * sizeof(pair)));
        int counter = 0;
        for (auto &&[key, val] : org_map)
        {
            data[counter] = {key, val};
            counter++;
        }
    }

    ~gpu_map()
    {
#ifndef __CUDACC__
        cudaFree(data);
        data = nullptr;
        size = 0;
#endif
    }

    // __device__ int find(MINIFE_GLOBAL_ORDINAL x)
    // {
    //     int low = 0;
    //     int high = size - 1;
    //     // Repeat until the pointers low and high meet each other
    //     while (low <= high)
    //     {
    //         int mid = low + (high - low) / 2;

    //         if (data[mid] == x)
    //             return mid;

    //         if (data[mid] < x)
    //             low = mid + 1;

    //         else
    //             high = mid - 1;
    //     }

    //     return -1;
    // }

    __device__ pair *lower_bound(const K &value) const
    {
        pair *it;
        ptrdiff_t count, step;
        count = size;
        auto first = data;

        while (count > 0)
        {
            it = first;
            step = count / 2;
            it += step;

            if (it->first < value)
            {
                first = ++it;
                count -= step + 1;
            }
            else
                count = step;
        }

        return first;
    }

    __device__ pair *find(const K &x) const
    {
        pair *res = lower_bound(x);
        if (res == end())
        {
            return end();
        }
        else if (res->first == x)
        {
            return res;
        }
        return end();
    }
    __device__ pair *begin() const
    {
        return data;
    }
    __device__ pair *end() const
    {
        return data + size;
    }
};

struct gpu_vector
{
    typedef MINIFE_SCALAR ScalarType;
    typedef MINIFE_GLOBAL_ORDINAL GlobalOrdinalType;
    typedef MINIFE_LOCAL_ORDINAL LocalOrdinalType;

    const GlobalOrdinalType startIndex;
    const LocalOrdinalType local_size;
    MINIFE_SCALAR *MINIFE_RESTRICT coefs;

    template <typename VectorType>
    gpu_vector(const VectorType &other) : startIndex(other.startIndex), local_size(other.local_size), coefs(other.coefs) {}
};

struct gpu_matrix
{

    typedef MINIFE_SCALAR ScalarType;
    typedef MINIFE_GLOBAL_ORDINAL GlobalOrdinalType;

    bool has_local_indices;
    MINIFE_GLOBAL_ORDINAL *rows;
    size_t rows_size;
    MINIFE_LOCAL_ORDINAL *row_offsets;
    MINIFE_LOCAL_ORDINAL *row_offsets_external;
    MINIFE_GLOBAL_ORDINAL *packed_cols;
    MINIFE_SCALAR *packed_coefs;
    MINIFE_LOCAL_ORDINAL num_cols;

    static __device__ MINIFE_GLOBAL_ORDINAL *lower_bound(MINIFE_GLOBAL_ORDINAL *first, MINIFE_GLOBAL_ORDINAL *last, const MINIFE_GLOBAL_ORDINAL &value)
    {
        MINIFE_GLOBAL_ORDINAL *it;
        ssize_t count, step;
        count = last - first;

        while (count > 0)
        {
            it = first;
            step = count / 2;
            it += step;

            if (*it < value)
            {
                first = ++it;
                count -= step + 1;
            }
            else
                count = step;
        }

        return first;
    }

    __device__ void get_row_pointers(MINIFE_GLOBAL_ORDINAL row, size_t &row_length,
                                     MINIFE_GLOBAL_ORDINAL *&cols,
                                     MINIFE_SCALAR *&coefs)
    {
        ptrdiff_t local_row = -1;
        // first see if we can get the local-row index using fast direct lookup:
        if (rows_size >= 1)
        {
            ptrdiff_t idx = row - rows[0];
            if (idx < rows_size && rows[idx] == row)
            {
                local_row = idx;
            }
        }

        // if we didn't get the local-row index using direct lookup, try a
        // more expensive binary-search:
        if (local_row == -1)
        {
            MINIFE_GLOBAL_ORDINAL *row_iter = gpu_matrix::lower_bound(rows, rows + rows_size, row);

            // if we still haven't found row, it's not local so jump out:
            if (row_iter == rows + rows_size || *row_iter != row)
            {
                row_length = 0;
                return;
            }

            local_row = row_iter - rows;
        }

        MINIFE_LOCAL_ORDINAL offset = row_offsets[local_row];
        row_length = row_offsets[local_row + 1] - offset;
        cols = &packed_cols[offset];
        coefs = &packed_coefs[offset];
    }
};