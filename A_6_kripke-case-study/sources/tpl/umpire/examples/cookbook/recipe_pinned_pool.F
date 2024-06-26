!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
! project contributors. See the COPYRIGHT file for details.
!
! SPDX-License-Identifier: (MIT)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program umpire_f_pinned_pool
      use umpire_mod
      implicit none
      logical ok

      integer(C_INT), pointer, dimension(:) :: array(:)
      type(UmpireAllocator) base_allocator
      type(UmpireAllocator) pinned_pool
      type(UmpireResourceManager) rm

      ! _sphinx_tag_tut_pinned_fortran_start
      rm = rm%get_instance()
      base_allocator = rm%get_allocator_by_name("PINNED")

      pinned_pool = rm%make_allocator_quick_pool("PINNED_POOL", &
                                           base_allocator, &
                                           512_8*1024_8, &
                                           1024_8)
      ! _sphinx_tag_tut_pinned_fortran_end

      call pinned_pool%allocate(array, [10])
end program umpire_f_pinned_pool
