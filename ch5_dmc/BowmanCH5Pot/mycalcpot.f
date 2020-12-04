      SUBROUTINE mycalcpot(x,v,nwalk)
      implicit real*8 (a-h,o-z)
      parameter (natom = 6)
      parameter (ndim = 3*natom)
      !dimension coord(ndim,nwalk)
	  integer, intent(in) :: nwalk
	  double precision, dimension (nwalk), intent(out) :: v
      double precision, dimension(nwalk,natom,3), intent(in) :: x
      emin = -40.65276470207075d0 + 2.9123825129317993d-007 !global surface
c     !print*,'enter number of walkers'
      !read(5,*)mwalker
      ! if (mwalker.gt.nwalk) then
         ! print*,'number of walkers is too high!'
         ! print*,'as currently implemented it must'
         ! print*,'not exceed ',nwalk
         ! stop
      !endif
! c     print*,'enter cartesian coordintes as'
! c     print*,'x1,y1,z1,...,xN,yN,zN for all'
! c     print*,'of the walkers'
      ! read(5,*)((coord(j,i),j=1,ndim),i=1,mwalker)
      do i = 1,nwalk
         ! ip = 0
         ! do j = 1,natom
            ! do k = 1,3
               ! ip = ip + 1
               ! x(j,k) = coord(ip,i)
            ! enddo
         ! enddo
	  call getpot(x,r,v(i))
         v(i) = v(i)-emin !vmin is in H
      enddo
      !write(6,*)(v(j),j=1,mwalker)
      !stop
      
	  END SUBROUTINE mycalcpot
