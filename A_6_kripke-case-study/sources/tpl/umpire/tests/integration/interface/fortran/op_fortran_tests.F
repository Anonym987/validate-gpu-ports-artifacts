! Generated by genumpireftests.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
! project contributors. See the COPYRIGHT file for details.
!
! SPDX-License-Identifier: (MIT)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module umpire_fortran_op_tests

  use iso_c_binding
  use fruit
  use umpire_mod
  implicit none

  contains



      subroutine test_copy_host_host
        use iso_c_binding

        type(UmpireResourceManager) rm
        type(UmpireAllocator) source_allocator
        type(UmpireAllocator) dest_allocator

        type(C_PTR) source
        type(C_PTR) dest

        rm = rm%get_instance()
        source_allocator = rm%get_allocator_by_name("HOST")
        dest_allocator = rm%get_allocator_by_name("HOST")

        source = source_allocator%allocate_pointer(1024_8)
        dest = dest_allocator%allocate_pointer(1024_8)

        call rm%copy(dest, source)

        call assert_true(.true.)

        call source_allocator%deallocate_pointer(source)
        call dest_allocator%deallocate_pointer(dest)

        call source_allocator%delete()
        call dest_allocator%delete()
      end subroutine test_copy_HOST_HOST

    

end module umpire_fortran_op_tests


program fortran_test
  use fruit
  use umpire_fortran_op_tests

  implicit none
  logical ok

  call init_fruit

  call test_copy_host_host

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
    call exit(1)
  endif
end program fortran_test

