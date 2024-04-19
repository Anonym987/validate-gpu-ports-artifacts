#pragma once

#include <aggpucontainers.hpp>
#include <set>
#include <map>
#include <simple_mesh_description.hpp>
#include <Hex8_box_utils.hpp>

namespace miniFE
{

    template <typename GlobalOrdinal>
    class simple_mesh_description_gpu
    {
    public:
        gpu_set<GlobalOrdinal> bc_rows_0;
        gpu_set<GlobalOrdinal> bc_rows_1;
        gpu_map<GlobalOrdinal, GlobalOrdinal> map_ids_to_rows;
        Box global_box;
        Box local_box;

        simple_mesh_description_gpu(const simple_mesh_description<GlobalOrdinal> &org) : bc_rows_0(org.bc_rows_0), bc_rows_1(org.bc_rows_1), map_ids_to_rows(org.map_ids_to_rows)
        {
            global_box = org.global_box;
            local_box = org.local_box;
        }

        __device__ GlobalOrdinal map_id_to_row(const GlobalOrdinal &id) const
        {
            auto iter = map_ids_to_rows.lower_bound(id);

            if (iter == map_ids_to_rows.end() || iter->first != id)
            {
                if (map_ids_to_rows.size > 0)
                {
                    --iter;
                }
                else
                {
                    printf("ERROR, failed to map id to row.\n");
                    return -99;
                }
            }

            if (iter->first == id)
            {
                return iter->second;
            }

            if (iter == map_ids_to_rows.begin() && iter->first > id)
            {
                printf("ERROR, id: %i, ids_to_rows.begin(): %i\n", id, iter->first);
                return -99;
            }

            GlobalOrdinal offset = id - iter->first;

            if (offset < 0)
            {
                printf("ERROR, negative offset in find_row_for_id for id=%i\n", id);
                return -99;
            }

            return iter->second + offset;
        }

    }; // class simple_mesh_description

    template <typename GlobalOrdinal, typename Scalar>
    __device__ __host__ void
    get_elem_nodes_and_coords_gpu(const simple_mesh_description_gpu<GlobalOrdinal> &mesh,
                                  GlobalOrdinal elemID,
                                  GlobalOrdinal *node_ords, Scalar *node_coords)
    {

        int global_nodes_x = mesh.global_box[0][1] + 1;
        int global_nodes_y = mesh.global_box[1][1] + 1;
        int global_nodes_z = mesh.global_box[2][1] + 1;

        if (elemID < 0)
        {
            // I don't think this can happen, but check for the sake of paranoia...
            assert(false && "get_elem_nodes_and_coords ERROR, negative elemID");
        }

        int elem_int_x, elem_int_y, elem_int_z;
        get_int_coords(elemID, global_nodes_x - 1, global_nodes_y - 1, global_nodes_z - 1,
                       elem_int_x, elem_int_y, elem_int_z);
        GlobalOrdinal nodeID = get_id<GlobalOrdinal>(global_nodes_x, global_nodes_y, global_nodes_z, elem_int_x, elem_int_y, elem_int_z);

#ifdef MINIFE_DEBUG_VERBOSE
        std::cout << "\nelemID: " << elemID << ", nodeID: " << nodeID << std::endl;
#endif
        get_hex8_node_ids(global_nodes_x, global_nodes_y, nodeID, node_ords);

        // Map node-IDs to rows because each processor may have a non-contiguous block of
        // node-ids, but needs a contiguous block of row-numbers:
#ifdef MINIFE_DEBUG_VERBOSE
        std::cout << "elem " << elemID << " nodes: ";
#endif
        for (int i = 0; i < Hex8::numNodesPerElem; ++i)
        {
#ifdef MINIFE_DEBUG_VERBOSE
            std::cout << node_ords[i] << " ";
#endif
            node_ords[i] = mesh.map_id_to_row(node_ords[i]);
        }
#ifdef MINIFE_DEBUG_VERBOSE
        std::cout << std::endl;
#endif

        int global_elems_x = mesh.global_box[0][1];
        int global_elems_y = mesh.global_box[1][1];
        int global_elems_z = mesh.global_box[2][1];

        Scalar ix, iy, iz;
        get_coords<GlobalOrdinal, Scalar>(nodeID, global_nodes_x, global_nodes_y, global_nodes_z,
                                          ix, iy, iz);
        Scalar hx = 1.0 / global_elems_x;
        Scalar hy = 1.0 / global_elems_y;
        Scalar hz = 1.0 / global_elems_z;
        get_hex8_node_coords_3d(ix, iy, iz, hx, hy, hz, node_coords);
#ifdef MINIFE_DEBUG_VERBOSE
        int offset = 0;
        for (int i = 0; i < Hex8::numNodesPerElem; ++i)
        {
            std::cout << "(" << node_coords[offset++] << "," << node_coords[offset++] << "," << node_coords[offset++] << ")";
        }
        std::cout << std::endl;
#endif
    }

} // namespace miniFE
