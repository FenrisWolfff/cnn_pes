c
c       THIS PROGRAM IS FOR VICTOR TO CHECK HIS CODES
C
      program checkcheck
      implicit real*8 (a-h,o-z)
      parameter (natom = 6)
      parameter (ndim = 3*natom)
      double precision, dimension(6,3) :: x
      emin = -40.65276470207075d0 + 2.9123825129317993d-007 !global surface
c     print*,'enter number of walkers'
c     print*,'enter cartesian coordintes as'
c     print*,'x1,y1,z1,...,xN,yN,zN for all'
c     print*,'of the walkers'
c     print*,'note the order is C, H1, H2, ...'
       open (unit=7,file='eq.dat',status='old')
       do ij = 1,3
         read(7,*)((x(i,j),j=1,3),i=1,natom) !make an array of coordinates that looks exactly like this
	call mycalcpot(x,v,1) !evaluate the potential at coordinates x (in a.u.
        print*,v,emin
        v = v-emin !vmin is in H
        print*,v,v*219474.6
      enddo
      stop
      end
